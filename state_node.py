from typing import Annotated, Any, Optional, TypedDict
from groq import BaseModel

from langgraph.graph.message import add_messages
from typing import Literal

class AppState:
    llm = None
    agent = None

class StateNode(TypedDict):
    messages:Annotated[list[Any],add_messages]
    next_action: Optional[str]

class RoutePlanner(BaseModel):
    next_action: Literal["chat","research","FINISH"]