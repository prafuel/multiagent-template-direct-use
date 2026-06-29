from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages


class CodeExecutionState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    start_time: float  # wall-clock start time (time.time()) for timeout detection
    code_snippet: str  # the generated Python code from the coder node
    approval_status: str  # "auto_approved" | "escalate" | "human_approved" | "aborted" | "retry"
    retry_feedback: str  # human feedback when status is "retry"
