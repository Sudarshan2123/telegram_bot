from typing import Annotated, Any, TypeDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages


class AppState:
    llm : ChatGoogleGenerativeAI = None
    agent = None

class StateNode(TypeDict):
    message:Annotated[list[Any],add_messages]