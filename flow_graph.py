from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage
from state_node import StateNode
from functools import partial

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant.")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}

def create_flow_graph(chatllm):
    graph = StateGraph(StateNode)
    graph.add_node("chat", partial(chat, chatllm=chatllm))
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()