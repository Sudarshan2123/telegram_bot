from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from state_node import StateNode

async def Orchesterator(state: StateNode,chatllm):
    """Orchestrator for the Telegram Agentic Bot."""
    messages = state["messages"][-1]
    system_msg = SystemMessage(content="You are a helpful assistant that provides concise user intent information.from the following intent : General Chat,insurance supporty")
    intent = await chatllm.ainvoke([system_msg, messages])
    return {"messages":[intent]}

async def Analyze_Photo(state: StateNode):
    """Analyze photo content."""
    messages = state.get("messages")[-1]
    system_msg = "You are an image analysis assistant. get all the content of the image in detail. for future Analysis"
    analysis = await state.agent.ainvoke({
        "messages": [system_msg, *messages]
    })
    return analysis["messages"][-1].content



def create_flow_graph()->StateGraph:
    """Create a flow graph for the Telegram Agentic Bot."""
    graph=StateGraph(StateNode)
    graph.add_node("Router",Orchesterator)
    graph.add_node("Analyze",Analyze_Photo)

    graph.add_edge(START, "Router")
    graph.add_edge("Router","Analyze")
    

    return graph,compile