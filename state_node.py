from typing import Annotated, Any, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages


class AppState:
    llm : ChatGoogleGenerativeAI = None
    agent = None

class StateNode(TypedDict):
    messages:Annotated[list[Any],add_messages]
    intent: Optional[str]

