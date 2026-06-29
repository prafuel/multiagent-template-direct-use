import logging

from langgraph.graph import StateGraph, START, END

from src.agents.code_execution.src.graph.states import CodeExecutionState
from src.agents.code_execution.src.graph.nodes import (
    coder_node,
    approval_agent_node,
    human_proxy_node,
    execute_code_node,
    abort_node,
    after_approval_router,
    after_human_proxy_router,
    after_coder_router,
)

logger = logging.getLogger("agent.code_execution")


def get_workflow():
    """Build and compile the Code Execution agent graph.

    Graph topology:
        START -> coder -> [safeguard check] -> approval_agent
                                            -> END (if safeguard aborted)
        approval_agent -> [auto_approved]   -> execute_code -> END
                       -> [escalate]        -> human_proxy
        human_proxy    -> [human_approved]  -> execute_code -> END
                       -> [retry]           -> coder (loop)
                       -> [aborted]         -> abort -> END
    """
    graph_builder = StateGraph(CodeExecutionState)

    # ── Nodes ──
    graph_builder.add_node("coder", coder_node)
    graph_builder.add_node("approval_agent", approval_agent_node)
    graph_builder.add_node("human_proxy", human_proxy_node)
    graph_builder.add_node("execute_code", execute_code_node)
    graph_builder.add_node("abort", abort_node)

    # ── Edges ──
    graph_builder.add_edge(START, "coder")

    # After coder: proceed to approval (or END if safeguard triggered)
    graph_builder.add_conditional_edges(
        "coder",
        after_coder_router,
        {"approval_agent": "approval_agent", "__end__": END},
    )

    # After approval_agent: auto_approved -> execute, escalate -> human
    graph_builder.add_conditional_edges(
        "approval_agent",
        after_approval_router,
        {"execute_code": "execute_code", "human_proxy": "human_proxy"},
    )

    # After human_proxy: approved -> execute, retry -> coder, aborted -> abort
    graph_builder.add_conditional_edges(
        "human_proxy",
        after_human_proxy_router,
        {"execute_code": "execute_code", "abort": "abort", "coder": "coder"},
    )

    # Terminal nodes
    graph_builder.add_edge("execute_code", END)
    graph_builder.add_edge("abort", END)

    agent_graph = graph_builder.compile()
    logger.info("Code Execution agent graph compiled successfully.")
    return agent_graph
