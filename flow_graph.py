import asyncio
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from state_node import RoutePlanner, StateNode
from functools import partial
from ddgs import DDGS

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="""You are a helpful vehicle insurance assistant.
Answer questions about vehicle insurance clearly and helpfully.""")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode, chatllm):
    messages = state["messages"]

    # check only last message for image
    last_msg = messages[-1]
    if isinstance(last_msg.content, list):
        if any(p.get("type") == "image_url" for p in last_msg.content if isinstance(p, dict)):
            print("Image detected → routing to analyse_photos")
            return {"next_action": "analyse_photos"}

    # ── Strip images from history before sending to Groq ──
    clean_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            text_parts = [p["text"] for p in msg.content if p.get("type") == "text"]
            if text_parts:
                clean_messages.append(HumanMessage(content=" ".join(text_parts)))
        else:
            clean_messages.append(msg)

    # ── Only send last 3 messages to keep tokens low ──
    recent = clean_messages[-3:]

    system_prompt = SystemMessage(content="""You are a routing supervisor.
Respond with ONLY one of these exact values for next_action:
- "research" → for insurance questions, plan queries, coverage, recommendations
- "chat" → for greetings, general conversation
- "FINISH" → when task is complete
Do NOT use any other values.""")

    planner = chatllm.with_structured_output(RoutePlanner)
    response = await planner.ainvoke([system_prompt] + recent)
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


async def Research(state: StateNode, chatllm, llm):
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

    # ── Only last 2 messages to keep tokens low ──
    recent_messages = text_messages[-2:]

    # ── Build query from last AI message (vehicle analysis) ──
    vehicle_context = "vehicle"
    for msg in reversed(recent_messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
            vehicle_context = msg.content[:150]
            break

    query = f"best vehicle insurance India 2025 {vehicle_context[:80]}"
    print(f"Research query: {query[:100]}")

    # ── DuckDuckGo search ──
    def search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=3))

    results = []
    try:
        results = await asyncio.to_thread(search)
        print(f"Search results found: {len(results)}")
    except Exception as e:
        print(f"Search error: {e}")

    context = "\n\n".join([
        f"• {r['title']}: {r['body'][:150]}"
        for r in results
    ]) if results else ""

    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert in India.
Recommend top 3 insurance plans based on the conversation.
{"Search Results:\n" + context if context else "Use your knowledge."}
For each: insurer name, premium (INR), 2 key benefits.""")

    # ← use llm (Scout 17B) for better output
    response = await llm.ainvoke([system_msg, *recent_messages])

    reply = response.content if isinstance(response.content, str) else str(response.content)
    reply = reply.strip()

    if not reply:
        reply = ("Top vehicle insurance in India:\n\n"
                 "1. HDFC ERGO — Comprehensive, INR 6,000–15,000/year\n"
                 "2. Bajaj Allianz — Good claims, INR 5,500–14,000/year\n"
                 "3. ICICI Lombard — Wide network, INR 6,500–16,000/year")

    print(f"Final reply: {reply[:200]}")
    return {"messages": [AIMessage(content=reply)]}


def route(state):
    action = state.get("next_action", "chat")
    if action not in ["chat", "research", "analyse_photos", "FINISH"]:
        print(f"Unknown route '{action}', defaulting to chat")
        return "chat"
    return action


def create_flow_graph(chatllm, llm): 
    graph = StateGraph(StateNode)

    graph.add_node("supervisor",     partial(Supervisor,     chatllm=chatllm))
    graph.add_node("chat",           partial(chat,           chatllm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_node("research",       partial(Research,       chatllm=chatllm, llm=llm))

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