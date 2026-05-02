from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from state_node import RoutePlanner, StateNode
from functools import partial

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant.")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode,llm):
    system_prompt = SystemMessage(content="""You are a routing supervisor.
        You MUST respond with ONLY one of these exact values for next_action:
        - "chat" → for any text message, greeting, or insurance question
        - "analyse_photos" → ONLY when the message contains an image or mentions analyzing a photo
        - "FINISH" → when the task is fully complete

        Do NOT use any other values. Do NOT use "assistant", "Rag", or anything else.""")
    text_only_messages = []
    for msg in state["messages"]:
        if isinstance(msg.content, list):  # multimodal message (has image)
            text_parts = [p["text"] for p in msg.content if p["type"] == "text"]
            text = " ".join(text_parts) or "User sent an image, analyze it."
            text_only_messages.append(HumanMessage(content=text))
        else:
            text_only_messages.append(msg)
    planner= llm.with_structured_output(RoutePlanner)

    response = await planner.ainvoke([system_prompt] + text_only_messages) 
    return {"next_action": response.next_action}


async def Analyse_photos(state: StateNode, llm):
    messages=state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant that analyze photos and provide insights.")
    response = await llm.ainvoke([system_msg, *messages])
    return {"messages": [response]}
    

def create_flow_graph(chatllm,llm):
    graph = StateGraph(StateNode)
    graph.add_node("chat", partial(chat, chatllm=chatllm))
    graph.add_node("supervisor", partial(Supervisor, llm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", 
        lambda state: state["next_action"],
        {
          "chat": "chat",
          "Analyse photos": "analyse_photos", 
          "FINISH": END
        }
    )
    graph.add_edge("analyse_photos",END)
    graph.add_edge("chat",END)
    return graph.compile()