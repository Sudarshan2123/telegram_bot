from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from state_node import RoutePlanner, StateNode
from functools import partial
from qdrant_client import QdrantClient


async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="""You are a helpful vehicle insurance assistant. 
    Answer questions about vehicle insurance clearly and helpfully.""")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode, chatllm):
    messages = state["messages"]

    # Detect image directly — no LLM needed
    for msg in messages:
        if isinstance(msg.content, list):
            if any(p.get("type") == "image_url" for p in msg.content if isinstance(p, dict)):
                print("Image detected → routing to analyse_photos")
                return {"next_action": "analyse_photos"}

    system_prompt = SystemMessage(content="""You are a routing supervisor.
Respond with ONLY one of these exact values for next_action:
- "rag" → for insurance questions, plan queries, coverage questions, recommendations
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


async def Rag(state: StateNode, chatllm, qdrant_client: QdrantClient):
    messages = state["messages"]

    # Build query from last few messages
    last_message = messages[-1]
    if isinstance(last_message.content, list):
        # extract text from multimodal message
        text_parts = [p["text"] for p in last_message.content if p.get("type") == "text"]
        query = " ".join(text_parts) or "vehicle insurance recommendation"
    else:
        query = last_message.content

    # also include second to last if it's an AI analysis of a photo
    if len(messages) >= 2:
        prev = messages[-2]
        if hasattr(prev, 'content') and isinstance(prev.content, str):
            query = f"{prev.content} {query}"

    print(f"RAG query: {query[:100]}")

    # Qdrant server-side embedding + search
    results = qdrant_client.query(
        collection_name="insurance",
        query_text=query,   # ← Qdrant embeds this server-side
        limit=3
    )

    if results:
        context = "\n\n".join([
            f"Plan: {r.metadata.get('plan_name', 'Unknown')}\n"
            f"Insurer: {r.metadata.get('insurer', 'Unknown')}\n"
            f"Premium: {r.metadata.get('premium_range', 'Unknown')}\n"
            f"Suitable For: {r.metadata.get('suitable_for', 'Unknown')}\n"
            f"Details: {r.document}"
            for r in results
        ])
    else:
        context = "No specific insurance plans found."

    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert.
Based on the retrieved insurance plans below, recommend the best option for the user.
Be specific about why each plan suits their needs and mention the premium range.

Retrieved Plans:
{context}
""")

    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


def route(state):
    action = state.get("next_action", "chat")
    if action not in ["chat", "rag", "analyse_photos", "FINISH"]:
        print(f"Unknown route '{action}', defaulting to chat")
        return "chat"
    return action


def create_flow_graph(chatllm, llm, qdrant_client):  # ← qdrant_client instead of retriever
    graph = StateGraph(StateNode)

    graph.add_node("supervisor",     partial(Supervisor,     chatllm=chatllm))
    graph.add_node("chat",           partial(chat,           chatllm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_node("rag",            partial(Rag,            chatllm=chatllm, qdrant_client=qdrant_client))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "chat": "chat",
            "rag": "rag",
            "analyse_photos": "analyse_photos",
            "FINISH": END
        }
    )

    graph.add_edge("chat", END)
    graph.add_edge("rag", END)
    graph.add_edge("analyse_photos", "rag")  # ← photo analysis → then RAG for plan recommendation

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)