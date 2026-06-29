from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class HumanProxyState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    start_time: float  # wall-clock start for timeout detection
    query: str  # the extracted question/request meant for the human
    decision: str  # "resolve" | "escalate"
    proxy_response: str  # the response from either the proxy LLM or the real human
    confidence: str  # "high" | "medium" | "low" — the proxy's self-assessed confidence
    escalation_reason: str  # why the proxy chose to escalate (empty if resolved)
