import asyncio
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from state_node import RoutePlanner, StateNode
from functools import partial
from duckduckgo_search import DDGS

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="""You are a helpful vehicle insurance assistant.
Answer questions about vehicle insurance clearly and helpfully.""")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode, chatllm):
    messages = state["messages"]

    last_msg = messages[-1]
    if isinstance(last_msg.content, list):
        if any(p.get("type") == "image_url" for p in last_msg.content if isinstance(p, dict)):
            print("Image detected → routing to analyse_photos")
            return {"next_action": "analyse_photos"}

    system_prompt = SystemMessage(content="""You are a routing supervisor.
Respond with ONLY one of these exact values for next_action:
- "research" → for insurance questions, plan queries, coverage questions, recommendations
- "chat" → for greetings, general conversation, non-insurance questions
- "FINISH" → when task is complete
Do NOT use any other values.""")

    planner = chatllm.with_structured_output(RoutePlanner)
    response = await planner.ainvoke([system_prompt] + messages)
    return {"next_action": response.next_action}


async def Analyse_photos(state: StateNode, llm):
    messages = state["messages"]
    system_msg = SystemMessage(content="""You are a vehicle insurance expert with vision capabilities.
Analyze the vehicle in the image and extract:
1. Vehicle type (sedan, SUV, hatchback, etc.)
2. Estimated make/model if visible
3. Any visible damage or condition issues
4. Age/condition of the vehicle

Provide a structured analysis that will help recommend the right insurance plan.""")
    response = await llm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Research(state: StateNode, chatllm):
    messages = state["messages"]

    # ── Strip images, keep AI analysis ──
    text_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_parts = [p["text"] for p in msg.content if p.get("type") == "text"]
            text_messages.append(HumanMessage(
                content=" ".join(text_parts) if text_parts else "[User sent a vehicle photo]"
            ))
        else:
            text_messages.append(msg)

    # ── Build search query ──
    query = "best vehicle insurance plans India"
    for msg in reversed(text_messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
            query = msg.content[:300] + " vehicle insurance India best plan"
            break

    print(f"Research query: {query[:100]}")

    def search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=3))

    try:
        results = await asyncio.to_thread(search)
        context = "\n\n".join([
            f"Title: {r['title']}\nSource: {r['href']}\nSummary: {r['body']}"
            for r in results
        ]) if results else ""
    except Exception as e:
        print(f"Search error: {e}")
        context = ""

    print(f"Search results found: {len(results) if results else 0}")

    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert in India.
Based on the search results and conversation below, recommend the best insurance plans.

{"Search Results:" + context if context else "No search results found. Use your knowledge to recommend plans."}

Provide:
1. Top 3 recommended insurance plans with insurer names
2. Premium ranges (in INR)
3. Key benefits of each plan
4. Why each plan suits this vehicle/situation
5. Claim process overview

Be specific, practical and helpful.""")

    response = await chatllm.ainvoke([system_msg, *text_messages])
    reply = response.content if isinstance(response.content, str) else str(response.content)
    reply = reply.strip() or "Sorry, I couldn't find recommendations right now. Please try again."

    print(f"Research reply: {reply[:200]}")
    return {"messages": [AIMessage(content=reply)]}


def route(state):
    action = state.get("next_action", "chat")
    if action not in ["chat", "research", "analyse_photos", "FINISH"]:
        print(f"Unknown route '{action}', defaulting to chat")
        return "chat"
    return action


def create_flow_graph(chatllm, llm):  # ← no qdrant_client
    graph = StateGraph(StateNode)

    graph.add_node("supervisor",     partial(Supervisor,     chatllm=chatllm))
    graph.add_node("chat",           partial(chat,           chatllm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_node("research",       partial(Research,       chatllm=chatllm))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "chat": "chat",
            "research": "research",
            "analyse_photos": "analyse_photos",
            "FINISH": END
        }
    )

    graph.add_edge("chat", END)
    graph.add_edge("research", END)
    graph.add_edge("analyse_photos", "research")  # photo → research

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)