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

    # No image — use Groq to route text
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
    system_msg = SystemMessage(content="You are a helpful assistant that analyze photos and provide insights.")
    response = await llm.ainvoke([system_msg, *messages])
    return {"messages": [response]}
    

def create_flow_graph(chatllm,llm):
    graph = StateGraph(StateNode)
    graph.add_node("chat", partial(chat, chatllm=chatllm))
    graph.add_node("supervisor", partial(Supervisor, chatllm=chatllm))
    graph.add_node("analyse_photos", partial(Analyse_photos, llm=llm))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", 
        lambda state: state["next_action"],
        {
          "chat": "chat",
          "analyse_photos": "analyse_photos", 
          "FINISH": END
        }
    )
    graph.add_edge("analyse_photos",END)
    graph.add_edge("chat",END)
    return graph.compile()