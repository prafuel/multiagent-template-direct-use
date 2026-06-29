def get_agent_description() -> str:
    """Return a short description of this agent for the orchestrator."""
    return (
        "Human Proxy Agent — Acts as an intelligent stand-in for the human user. "
        "When the system needs human input (decisions, feedback, approvals, "
        "clarifications), this agent evaluates whether the query is routine "
        "enough to handle autonomously or whether it truly requires real human "
        "attention. Routine queries are resolved instantly by the proxy; "
        "critical or ambiguous ones are escalated to the actual human. Use this "
        "agent whenever you need a human decision but want to avoid unnecessary "
        "interruptions."
    )
