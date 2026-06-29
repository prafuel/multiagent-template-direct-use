import os
import time
import logging

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage

from src.config import get_llm
from src.graph.states import OrchestratorState

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME_SECONDS", "120"))

logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _trim_memory(messages: list) -> list:
    """Keep system messages + last MEMORY_WINDOW human messages."""
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    human_count = 0
    cut_index = 0
    for i in range(len(non_system) - 1, -1, -1):
        if isinstance(non_system[i], HumanMessage):
            human_count += 1
            if human_count >= MEMORY_WINDOW:
                cut_index = i
                break

    if human_count < MEMORY_WINDOW:
        return system_msgs + non_system
    return system_msgs + non_system[cut_index:]


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator_graph(system_prompt: SystemMessage, all_tools: list):
    """Build and compile the orchestrator LangGraph."""

    tool_map = {t.name: t for t in all_tools}

    def orchestrator_node(state: OrchestratorState) -> dict:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(all_tools)

        iteration = state.get("iteration", 0) + 1
        messages = _trim_memory(state["messages"])

        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        messages = [system_prompt] + non_system

        elapsed = time.time() - state.get("start_time", time.time())

        logger.info("\n" + "="*60)
        logger.info("[ORCHESTRATOR]  Iteration %d/%d  |  Elapsed: %.1fs", iteration, MAX_ITERATIONS, elapsed)
        logger.info("="*60)

        if elapsed > MAX_EXECUTION_TIME:
            logger.warning("[ORCHESTRATOR]  TIMEOUT -- elapsed %.1fs > %ds. Forcing final answer.", elapsed, MAX_EXECUTION_TIME)
            return {
                "messages": [AIMessage(content="I've exceeded the time limit. Here is what I have so far.")],
                "iteration": iteration,
            }

        if iteration > MAX_ITERATIONS:
            logger.warning("[ORCHESTRATOR]  MAX ITERATIONS reached (%d). Forcing final answer.", MAX_ITERATIONS)
            return {
                "messages": [AIMessage(content="I've reached the maximum number of steps.")],
                "iteration": iteration,
            }

        response = llm_with_tools.invoke(messages)

        resp_type = 'TOOL CALL' if response.tool_calls else 'DIRECT ANSWER'
        logger.info("[ORCHESTRATOR]  Response: %s", resp_type)
        if response.tool_calls:
            for tc in response.tool_calls:
                logger.info("  | Tool:  %s", tc['name'])
                logger.info("  | Args:  %s", str(tc['args'])[:150])
        else:
            logger.info("  | Answer: %s...", response.content[:200])

        return {
            "messages": [response],
            "iteration": iteration,
        }

    def tool_executor_node(state: OrchestratorState) -> dict:
        last_msg = state["messages"][-1]
        tool_messages = []

        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            logger.info("-"*60)
            logger.info("[TOOL EXEC]  Running: %s", tool_name)
            logger.info("  | Args:  %s", str(tool_args)[:150])

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

        return {"messages": tool_messages}

    def after_orchestrator_router(state: OrchestratorState) -> Literal["tools", "__end__"]:
        last_msg = state["messages"][-1]
        iteration = state.get("iteration", 0)
        elapsed = time.time() - state.get("start_time", time.time())

        if iteration > MAX_ITERATIONS:
            return "__end__"
        if elapsed > MAX_EXECUTION_TIME:
            return "__end__"
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "__end__"

    # Build the graph
    graph_builder = StateGraph(OrchestratorState)

    graph_builder.add_node("orchestrator", orchestrator_node)
    graph_builder.add_node("tools", tool_executor_node)

    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_conditional_edges(
        "orchestrator",
        after_orchestrator_router,
        {"tools": "tools", "__end__": END},
    )
    graph_builder.add_edge("tools", "orchestrator")

    return graph_builder.compile()