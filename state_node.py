from typing import Annotated, Any, Optional, TypedDict
from groq import BaseModel
from pydantic import BaseModel, field_validator
from langgraph.graph.message import add_messages
from typing import Literal

class AppState:
    llm = None
    agent = None

class StateNode(TypedDict):
    messages: Annotated[list, add_messages]
    next_action: str
    vehicle_context: str 

class RoutePlanner(BaseModel):
    next_action: Literal["research", "chat", "analyse_photos", "FINISH"]
 
    @field_validator("next_action", mode="before")
    @classmethod
    def normalize(cls, v):
        """Normalize LLM output to valid route values."""
        if not isinstance(v, str):
            return "chat"
        v = v.strip().lower()
        mapping = {
            "research":       "research",
            "do_research":    "research",
            "insurance":      "research",
            "chat":           "chat",
            "greet":          "chat",
            "general":        "chat",
            "analyse_photos": "analyse_photos",
            "analyze_photos": "analyse_photos",
            "photo":          "analyse_photos",
            "image":          "analyse_photos",
            "finish":         "FINISH",
            "done":           "FINISH",
            "end":            "FINISH",
        }
        return mapping.get(v, "chat")