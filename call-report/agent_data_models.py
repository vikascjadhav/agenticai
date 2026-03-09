
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import List
from pydantic import BaseModel


class ActionItem(BaseModel):
    description: str
    owner: str
    due_date: str


class ActionItemResult(BaseModel):
    description: str
    owner: str
    due_date: str
    status: str


class AgentState(TypedDict):
    text: str
    summary: str
    proposed_action_items: list[ActionItem]
    approved_action_items: list[ActionItem]
    created_tasks: list[ActionItemResult]
    review_index_of_action_items: int
    messages: Annotated[list[BaseMessage], add_messages]


class ActionItemRequest(BaseModel):
    proposed_action_items: List[ActionItem]

class ToolResponse(BaseModel):
    status: str
    created_tasks: List[ActionItemResult]
    count: int
