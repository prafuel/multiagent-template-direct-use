from langchain_core.messages import SystemMessage


def build_orchestrator_prompt(agents: dict, tools: list, agent_tools: list) -> SystemMessage:
    """Build the orchestrator system prompt dynamically from discovered agents
    and available common tools."""

    # Agent descriptions
    agent_lines = []
    for name, info in agents.items():
        agent_lines.append(f"  • delegate_to_{name}: {info['description']}")
    agents_block = "\n".join(agent_lines) if agent_lines else "  (No agents discovered)"

    # Common tool descriptions
    common_lines = []
    for t in tools:
        desc = t.description.split("\n")[0]
        common_lines.append(f"  • {t.name}: {desc}")
    common_block = "\n".join(common_lines) if common_lines else "  (No common tools)"

    return SystemMessage(content=f"""\
You are the Orchestrator — the central coordinator of a multi-agent system.
Your job is to understand the user's request and delegate work to the
appropriate sub-agent(s) or use common tools directly.

═══════════════════════════════════════════════════════════════
AVAILABLE SUB-AGENTS (use delegate_to_<name> tools):
{agents_block}
═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
COMMON TOOLS (use directly):
{common_block}
═══════════════════════════════════════════════════════════════

WORKFLOW RULES:
1. Analyze the user's request carefully.
2. If you need clarification, use ask_human directly.
3. If the task requires a specific agent's expertise, delegate using
   the appropriate delegate_to_<agent> tool with a clear task description.
4. You can delegate to multiple agents sequentially (e.g., plan first,
   then write a blog).
5. You can also use file tools directly to read/write files.
6. After receiving results from sub-agents, synthesize and present
   the final answer to the user.
7. Be concise and informative in your responses.

IMPORTANT:
- Do NOT try to do specialized work yourself — delegate to agents.
- When delegating, provide a CLEAR and DETAILED task description.
- If a user's request spans multiple agents, break it down and
  delegate to each one sequentially.
""")
