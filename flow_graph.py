from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from state_node import RoutePlanner, StateNode
from functools import partial

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant.")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode, chatllm):
    messages = state["messages"]
    
    # Check if any message contains an image — route directly, no LLM needed
    for msg in messages:
        if isinstance(msg.content, list):
            if any(p.get("type") == "image_url" for p in msg.content if isinstance(p, dict)):
                print("Image detected → routing to analyse_photos")
                return {"next_action": "analyse_photos"}  # skip LLM entirely

    system_prompt = SystemMessage(content="""You are a routing supervisor.
    Respond with ONLY one of these exact values for next_action:
    - "chat" → for text messages, greetings, insurance questions
    - "FINISH" → when task is complete
    Do NOT use any other values.""")

    planner = chatllm.with_structured_output(RoutePlanner)
    response = await planner.ainvoke([system_prompt] + messages)
    return {"next_action": response.next_action}


async def Analyse_photos(state: StateNode, llm):
    messages=state["messages"]
    last_message = messages[-1].content
    system_msg = SystemMessage(content="You are a helpful assistant that analyze photos and provide insights.")
    response = await llm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Rag(state: StateNode, chatllm, retriever):
    messages = state["messages"]
    
    # Build focused query from state fields
    query_parts = ["vehicle insurance plan"]
    if state.get("vehicle_type"):
        query_parts.append(f"vehicle: {state['vehicle_type']}")
    if state.get("driver_age"):
        query_parts.append(f"driver age: {state['driver_age']}")
    if state.get("damage_type"):
        query_parts.append(f"damage: {state['damage_type']}")
    if state.get("budget"):
        query_parts.append(f"budget: {state['budget']}")
    
    # fallback — use last user message directly
    query = ", ".join(query_parts) if len(query_parts) > 1 else messages[-1].content
    
    print(f"RAG query: {query}")
    
    # retrieve from Qdrant
    docs = await retriever.ainvoke(query)
    
    context = "\n\n".join([
        f"Plan: {doc.metadata.get('plan_name', 'Unknown')}\n"
        f"Insurer: {doc.metadata.get('insurer', 'Unknown')}\n"
        f"Premium: {doc.metadata.get('premium_range', 'Unknown')}\n"
        f"Details: {doc.page_content}"
        for doc in docs
    ])
    
    system_msg = SystemMessage(content=f"""You are a vehicle insurance expert.
Based on the retrieved insurance plans below, recommend the best option for the user.
Be specific about why each plan suits their needs.

Retrieved Plans:
{context}
""")
    
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}




def create_flow_graph(chatllm,llm,retriever):
    graph = StateGraph(StateNode)
    graph.add_node("chat", partial(chat, chatllm=chatllm))
    graph.add_node("supervisor", partial(Supervisor, chatllm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_node("rag", partial(Rag, chatllm=chatllm),retriever=retriever)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", 
        lambda state: state["next_action"],
        {
          "chat": "chat",
          "rag": "rag",
          "analyse_photos": "analyse_photos", 
          "FINISH": END
        }
    )
    graph.add_edge("analyse_photos","supervisor")
    graph.add_edge("chat",END)
    graph.add_edge("rag",END)
    return graph.compile()