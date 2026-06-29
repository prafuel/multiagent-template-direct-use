from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    start_time: float
