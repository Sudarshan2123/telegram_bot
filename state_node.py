from typing import Annotated, Any, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from typing import Literal

class AppState:
    llm : ChatGoogleGenerativeAI = None
    agent = None

class StateNode(TypedDict):
    messages:Annotated[list[Any],add_messages]
    next_action: Optional[str]

class RoutePlanner(TypedDict):
    next_action: Literal["Analyse photos","chat","Rag","assistant","FINISH"]