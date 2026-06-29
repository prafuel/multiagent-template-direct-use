# ====================== below code is just for template ===============================

import os
import time
import logging
from langchain_core.messages import ( 
    SystemMessage, 
    ToolMessage, 
    HumanMessage, 
    AIMessage
)

from typing import Literal

from src.config import get_llm
from src.agents.planning.get_tools import get_tools
from src.agents.planning.src.graph.states import AgentState
from src.agents.planning.src.graph.prompts import get_planner_prompt, SYSTEM_PROMPT

llm = get_llm()

all_tools = get_tools()

tool_map = {t.name: t for t in all_tools}

llm_with_tools = llm.bind_tools(all_tools)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))  # keep last N message-pairs
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME_SECONDS", "60"))

logger = logging.getLogger("agent.safeguards")


def _trim_memory(messages: list) -> list:
    """Keep the system message(s) + the last MEMORY_WINDOW user messages and
    all associated AI / tool messages between them.

    We traverse the message list backwards and count HumanMessages.  Once we
    have seen MEMORY_WINDOW HumanMessages we stop and keep everything from
    that point onwards.  This ensures that multi-step tool-call chains
    (AIMessage → ToolMessage → AIMessage) are never orphaned.
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    # Count HumanMessages from the end
    human_count = 0
    cut_index = 0  # index into non_system where we start keeping
    for i in range(len(non_system) - 1, -1, -1):
        if isinstance(non_system[i], HumanMessage):
            human_count += 1
            if human_count >= MEMORY_WINDOW:
                cut_index = i
                break

    # If we never hit the limit, keep everything
    if human_count < MEMORY_WINDOW:
        return system_msgs + non_system

    trimmed = non_system[cut_index:]
    return system_msgs + trimmed


def planner_node(state: AgentState) -> dict:
    """Generate a step-by-step plan WITHOUT calling any tools."""
    messages = _trim_memory(state["messages"])

    # Replace any existing system prompt with the planner prompt
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    planner_messages = [get_planner_prompt(all_tools)] + non_system

    logger.info("\n" + "="*60)
    logger.info("[PLANNER]  Generating step-by-step plan...")
    logger.info("  | Messages in context: %d", len(planner_messages))
    logger.info("="*60)

    # Invoke LLM WITHOUT tools bound so it can only produce text
    plan_response = llm.invoke(planner_messages)
    plan_text = plan_response.content.strip()

    logger.info("[PLANNER]  Plan generated:")
    logger.info("-"*40)
    logger.info("%s", plan_text)
    logger.info("-"*40)

    return {
        "plan": plan_text,
    }


def agent_node(state: AgentState) -> dict:
    """Call the LLM with tools, guided by the plan from the planner node."""
    iteration = state.get("iteration", 0) + 1
    messages = _trim_memory(state["messages"])

    # Build the system context: system prompt + injected plan
    plan_text = state.get("plan", "")
    plan_injection = f"\n\n--- CURRENT PLAN ---\n{plan_text}\n--- END PLAN ---"
    system_with_plan = SystemMessage(content=SYSTEM_PROMPT.content + plan_injection)

    # Replace any existing system messages with the plan-augmented one
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    messages = [system_with_plan] + non_system

    elapsed = time.time() - state.get("start_time", time.time())

    logger.info("\n" + "="*60)
    logger.info("[PLANNING AGENT]  Iteration %d/%d  |  Elapsed: %.1fs / %ds", iteration, MAX_ITERATIONS, elapsed, MAX_EXECUTION_TIME)
    logger.info("  | Messages in context: %d", len(messages))
    logger.info("="*60)

    # ── Safeguard: Wall-clock timeout ──
    if elapsed > MAX_EXECUTION_TIME:
        logger.warning("[PLANNING AGENT]  TIMEOUT -- elapsed %.1fs > %ds. Forcing final answer.", elapsed, MAX_EXECUTION_TIME)
        return {
            "messages": [AIMessage(content=f"I've exceeded the maximum execution time ({MAX_EXECUTION_TIME}s). Based on what I know so far, here is my best answer. Please try again if you need more detail.")],
            "iteration": iteration,
        }

    # ── Safeguard: Max iterations (loop detection) ──
    if iteration > MAX_ITERATIONS:
        logger.warning("[PLANNING AGENT]  MAX ITERATIONS reached (%d). Possible loop detected. Forcing final answer.", MAX_ITERATIONS)
        return {
            "messages": [AIMessage(content="I've reached the maximum number of reasoning steps -- this may indicate a loop. Based on what I know so far, I'm unable to provide a complete answer. Please try rephrasing your question.")],
            "iteration": iteration,
        }

    response = llm_with_tools.invoke(messages)

    resp_type = 'TOOL CALL' if response.tool_calls else 'DIRECT ANSWER'
    logger.info("[PLANNING AGENT]  Response: %s", resp_type)
    if response.tool_calls:
        for tc in response.tool_calls:
            logger.info("  | Tool:  %s", tc['name'])
            logger.info("  | Args:  %s", tc['args'])
    else:
        logger.info("  | Answer: %s...", response.content[:200])

    return {
        "messages": [response],
        "iteration": iteration,
    }


def tool_executor_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    last_msg = state["messages"][-1]
    tool_messages = []

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        logger.info("-"*60)
        logger.info("[TOOL EXEC]  Running: %s", tool_name)
        logger.info("  | Args:  %s", tool_args)

        try:
            result = tool_map[tool_name].invoke(tool_args)
            result_str = str(result)
            logger.info("  | Status: SUCCESS")
            logger.info("  | Result: %s", result_str[:300])

            tool_messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"])
            )
        except Exception as e:
            error_msg = f"TOOL ERROR ({tool_name}): {type(e).__name__}: {e}"
            logger.error("  | Status: FAILED")
            logger.error("  | Error:  %s", error_msg)
            tool_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tc["id"])
            )

    return {
        "messages": tool_messages,
    }


# ─────────────────────────────────────────────────────────────────────────────────
#  ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────────

def after_agent_router(state: AgentState) -> Literal["tools", "__end__"]:
    """After the agent node: either call tools or end.

    Safeguards checked:
    1. MAX_ITERATIONS — prevents infinite agent↔tool loops.
    2. MAX_EXECUTION_TIME — prevents runaway wall-clock usage.
    """
    last_msg = state["messages"][-1]
    iteration = state.get("iteration", 0)
    elapsed = time.time() - state.get("start_time", time.time())

    if iteration > MAX_ITERATIONS:
        logger.warning("Router: loop safeguard -- iteration %d, routing to END", iteration)
        return "__end__"

    if elapsed > MAX_EXECUTION_TIME:
        logger.warning("Router: timeout safeguard -- elapsed %.1fs, routing to END", elapsed)
        return "__end__"

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return "__end__"