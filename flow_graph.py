import os

import httpx
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


async def Rag(state: StateNode, chatllm, qdrant_client):
    messages = state["messages"]

    # ── Strip ONLY image content from HumanMessages, keep AI analysis ──
    text_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            # HumanMessage with image — extract text parts only
            text_parts = [p["text"] for p in msg.content if p.get("type") == "text"]
            if text_parts:
                text_messages.append(HumanMessage(content=" ".join(text_parts)))
            # if only image with no text, replace with a placeholder
            else:
                text_messages.append(HumanMessage(content="[User sent a vehicle photo]"))
        else:
            text_messages.append(msg)

    # ── Build query from AI analysis (last AI message) ──
    query = "vehicle insurance recommendation"
    for msg in reversed(text_messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
            query = msg.content[:500]  
            break

    print(f"RAG query: {query[:100]}")

    # ── Fetch plans from Qdrant ──
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{os.getenv('QDRANT_URL')}/collections/insurance/points/scroll",
            headers={
                "api-key": os.getenv("QDRANT_API_KEY"),
                "Content-Type": "application/json"
            },
            json={
                "limit": 5,
                "with_payload": True,
                "with_vector": False
            }
        )
        data = response.json()

    points = data.get("result", {}).get("points", [])

    if points:
        context = "\n\n".join([
            f"Plan: {p['payload'].get('plan_name', 'Unknown')}\n"
            f"Insurer: {p['payload'].get('insurer', 'Unknown')}\n"
            f"Premium: {p['payload'].get('premium_range', 'Unknown')}\n"
            f"Suitable For: {p['payload'].get('suitable_for', 'Unknown')}\n"
            f"Details: {p['payload'].get('document', p['payload'].get('page_content', ''))}"
            for p in points
        ])
    else:
        context = "No insurance plans found."

    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert.
A vehicle photo was analyzed and the analysis is included in the conversation below.
Based on the vehicle analysis AND the insurance plans provided, recommend the most suitable plan.
Explain specifically why the plan matches the vehicle's condition and type.

Available Insurance Plans:
{context}
""")

    response = await chatllm.ainvoke([system_msg, *text_messages])
    print(f"RAG response type: {type(response.content)}")
    print(f"RAG response content: {str(response.content)[:200]}")
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