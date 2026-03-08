
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List
from pydantic import BaseModel

class AgentState(TypedDict):
    text: str
    summary: str
    action_items: list
    messages: Annotated[list[BaseMessage], add_messages]
    
class ActionItem(BaseModel):
    description: str
    owner: str
    due_date: str

class ActionItemRequest(BaseModel):
    action_items: List[ActionItem]
    
class ActionItemResult(BaseModel):
    description: str
    owner: str
    due_date: str
    status: str
    
class ToolResponse(BaseModel):
    status: str
    created_tasks: List[ActionItemResult]
    count: int
