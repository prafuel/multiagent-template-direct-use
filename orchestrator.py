'''
orchestrator.py — Dynamic Multi-Agent Orchestrator

Enhanced with additional logging for detailed tracing of execution flow.
''' 

import os
import time
import importlib
import logging

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from src.get_tools import get_tools
from src.graph.prompts import build_orchestrator_prompt
from src.graph.nodes import build_orchestrator_graph

load_dotenv()

# ── Logging configuration ────────────────────────────────────────────────────
# Show execution traces (planning, tool calls, results) in the terminal.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Silence noisy third‑party libraries
for _lib in (
    "httpx",
    "groq",
    "httpcore",
    "urllib3",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_groq",
    "langsmith",
):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger("orchestrator")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "src", "agents")

# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC AGENT DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_agents() -> dict:
    """Scan src/agents/ for sub‑agent folders that expose:
       - agent_prompt.get_agent_description()
       - get_workflow.get_workflow()

    Returns a dict: { agent_name: { "description": str, "workflow_fn": callable } }
    """
    logger.debug("Starting agent discovery in %s", AGENTS_DIR)
    agents = {}

    if not os.path.isdir(AGENTS_DIR):
        logger.warning("Agents directory not found: %s", AGENTS_DIR)
        return agents

    for entry in sorted(os.listdir(AGENTS_DIR)):
        agent_path = os.path.join(AGENTS_DIR, entry)
        if not os.path.isdir(agent_path):
            logger.debug("Skipping non‑directory entry: %s", entry)
            continue

        # Skip __pycache__ and hidden dirs
        if entry.startswith("_") or entry.startswith("."):
            logger.debug("Skipping hidden or special directory: %s", entry)
            continue

        logger.debug("Attempting to import agent modules for: %s", entry)
        prompt_module_name = f"src.agents.{entry}.agent_prompt"
        workflow_module_name = f"src.agents.{entry}.get_workflow"

        try:
            prompt_mod = importlib.import_module(prompt_module_name)
            workflow_mod = importlib.import_module(workflow_module_name)
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning("Skipping agent '%s' due to import error: %s", entry, e)
            continue

        # Validate required callables
        if not hasattr(prompt_mod, "get_agent_description"):
            logger.warning("Skipping agent '%s': missing get_agent_description()", entry)
            continue
        if not hasattr(workflow_mod, "get_workflow"):
            logger.warning("Skipping agent '%s': missing get_workflow()", entry)
            continue

        agents[entry] = {
            "description": prompt_mod.get_agent_description(),
            "workflow_fn": workflow_mod.get_workflow,
        }
        logger.info("Discovered agent: %s", entry)

    logger.debug("Agent discovery completed. Total agents found: %d", len(agents))
    return agents

# ─────────────────────────────────────────────────────────────────────────────
#  AGENT‑AS‑TOOL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_tools(agents: dict) -> list:
    """Wrap each discovered agent's workflow as a LangChain Tool.

    Each tool, when invoked, compiles the agent graph and runs
    it with the provided task/message.
    """
    logger.debug("Building tool wrappers for %d agents", len(agents))
    agent_tools = []

    for agent_name, agent_info in agents.items():
        # Capture variables in a closure to avoid late binding issues
        def _make_tool(name, info):
            description = (
                f"Delegate a task to the {name} agent. {info['description']} "
                f"Args: task -- A clear description of what you want this agent to do."
            )

            @tool(f"delegate_to_{name}", description=description)
            def agent_tool(task: str) -> str:
                """Delegate a task to a sub‑agent and log the interaction."""
                logger.info("\n" + "=" * 60)
                logger.info("[DELEGATING] -> %s agent", name)
                logger.info("  | Task (truncated): %s", task[:200])
                logger.info("=" * 60)

                try:
                    logger.debug("Compiling workflow for agent %s", name)
                    workflow = info["workflow_fn"]()
                    logger.debug("Invoking workflow for agent %s", name)
                    result = workflow.invoke({
                        "messages": [HumanMessage(content=task)],
                        "iteration": 0,
                        "start_time": time.time(),
                    })
                    logger.debug("Workflow completed for agent %s", name)

                    # Extract the last AIMessage as the result
                    ai_messages = [
                        m for m in result["messages"] if isinstance(m, AIMessage) and m.content
                    ]
                    if ai_messages:
                        response = f"[{name} agent result]: {ai_messages[-1].content}"
                        logger.info("[RESULT] %s", response)
                        return response
                    else:
                        msg = f"[{name} agent]: Task completed but no text response generated."
                        logger.info(msg)
                        return msg
                except Exception as e:
                    err_msg = f"[{name} agent ERROR]: {type(e).__name__}: {e}"
                    logger.error(err_msg)
                    return err_msg

            return agent_tool

        tool_wrapper = _make_tool(agent_name, agent_info)
        agent_tools.append(tool_wrapper)
        logger.debug("Created tool wrapper: delegate_to_%s", agent_name)

    logger.debug("All agent tool wrappers built (%d total)", len(agent_tools))
    return agent_tools

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Multi‑Agent Orchestrator")
    print("\nMULTI-AGENT ORCHESTRATOR")
    print("=" * 40)

    # 1. Discover agents
    print("\nDiscovering agents...")
    agents = discover_agents()

    if not agents:
        print(
            "No agents discovered. Ensure agents are in src/agents/ with "
            "agent_prompt.py and get_workflow.py."
        )
        logger.warning("No agents found – exiting.")
        return

    print(f"Discovered {len(agents)} agent(s): {', '.join(agents.keys())}")
    logger.info("Discovered agents: %s", ", ".join(agents.keys()))

    # 2. Build agent tools + common tools
    agent_tools = build_agent_tools(agents)
    tools = get_tools()
    all_tools = agent_tools + tools

    print(f"Tools available: {len(all_tools)}")
    for t in all_tools:
        print(f"  - {t.name}")
    logger.debug("Tool list prepared with %d tools", len(all_tools))

    # 3. Build orchestrator prompt
    logger.debug("Building orchestrator system prompt")
    system_prompt = build_orchestrator_prompt(agents, tools, agent_tools)

    # 4. Compile orchestrator graph
    logger.debug("Compiling orchestrator graph")
    orchestrator_graph = build_orchestrator_graph(system_prompt, all_tools)
    print("Orchestrator ready.\n")
    logger.info("Orchestrator graph compiled and ready for interaction")

    # 5. Interactive loop
    separator = "-" * 40
    print(separator)
    print("Type your request below. Type 'quit' or 'exit' to stop.")
    print(separator)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            logger.info("User terminated session via interrupt")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            logger.info("User exited the orchestrator loop")
            break

        logger.info("User input received: %s", user_input)
        # Run the orchestrator
        logger.debug("Invoking orchestrator graph for user request")
        result = orchestrator_graph.invoke({
            "messages": [HumanMessage(content=user_input)],
            "iteration": 0,
            "start_time": time.time(),
        })

        # Extract the final answer
        ai_messages = [
            m for m in result["messages"] if isinstance(m, AIMessage) and m.content
        ]
        if ai_messages:
            response = ai_messages[-1].content
            print("\n" + separator)
            print("Orchestrator:", response)
            print(separator)
            logger.info("Orchestrator response sent to user")
        else:
            print("\nNo response generated.")
            logger.warning("Orchestrator returned no AIMessage")

    logger.info("Orchestrator session ended")

if __name__ == "__main__":
    main()
