from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage
from state_node import RoutePlanner, StateNode
from functools import partial

async def chat(state: StateNode, chatllm):
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant.")
    response = await chatllm.ainvoke([system_msg, *messages])
    return {"messages": [response]}


async def Supervisor(state: StateNode,llm):
    system_prompt = SystemMessage(content="You are a helpful Supervisor of a team :Analyse photos,researcher,Rag,assistant"
                                  "Based on the user query, you will decide which team member to call and what information to provide them."
                                  "if the task is completed respond with FINISH.")

    planner= llm.with_structured_output(RoutePlanner)

    response = await planner.ainvoke([{"role": "system", "content": system_prompt}] + state["messages"])
    return {"next_action": response.next_action}


async def Analyse_photos(state: StateNode, llm):
    messages=state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant that analyze photos and provide insights.")
    response = await llm.ainvoke([system_msg, *messages])
    return {"messages": [response]}
    

def create_flow_graph(chatllm,llm):
    graph = StateGraph(StateNode)
    graph.add_node("chat", partial(chat, chatllm=chatllm))
    graph.add_node("supervisor", partial(Supervisor, llm=llm))
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