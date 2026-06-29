def get_agent_description() -> str:
    """Return a short description of this agent for the orchestrator."""
    return (
        "Code Execution Agent — Generates Python code for a given task, runs it "
        "through an LLM-based safety review (auto-approving safe code, escalating "
        "risky code to the human), and executes the approved code locally. Supports "
        "retry with feedback and abort. Use this agent when the user wants to write "
        "AND run code."
    )
