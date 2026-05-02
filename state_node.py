from typing import Annotated, Any, Optional, TypedDict
from groq import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from typing import Literal

class AppState:
    llm : ChatGoogleGenerativeAI = None
    agent = None
    retriever = None

class StateNode(TypedDict):
    messages:Annotated[list[Any],add_messages]
    next_action: Optional[str]
    driver_age: Optional[int] = None
    vehicle_type: Optional[str] = None
    damage_type: Optional[str] = None
    budget: Optional[str] = None
    retrieved_docs: Optional[list] = None

class RoutePlanner(BaseModel):
    next_action: Literal["chat","rag","FINISH"]