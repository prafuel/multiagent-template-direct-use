from src.agents.planning.get_tools import get_tools

# here will get whole info about, what are the capabilities for this agent


def get_agent_description() -> str:
    """Return a short description of this agent for the orchestrator."""
    return (
        "Planning Agent — Analyzes user queries and produces step-by-step plans "
        "describing which tools to call and in what order. Has access to internet "
        "search (DuckDuckGo), Wikipedia lookup, and human-in-the-loop for "
        "clarification. Use this agent when you need to research a topic, gather "
        "information, or create a structured plan before taking action."
    )


def get_prompt() -> str:
    tool_info = {f"{tool.name} : {tool.description}" for tool in get_tools()}

    prompt = f"""This is planning agent and it has access to following tools {tool_info}"""

    return prompt