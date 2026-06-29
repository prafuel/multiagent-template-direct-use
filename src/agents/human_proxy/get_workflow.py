import logging

from langgraph.graph import StateGraph, START, END

from src.agents.human_proxy.src.graph.states import HumanProxyState
from src.agents.human_proxy.src.graph.nodes import (
    evaluator_node,
    actual_human_node,
    after_evaluator_router,
)

logger = logging.getLogger("agent.human_proxy")


def get_workflow():
    """Build and compile the Human Proxy agent graph.

    Graph topology:
        START -> evaluator
        evaluator -> [resolve]   -> END  (proxy answered autonomously)
                  -> [escalate]  -> actual_human -> END
    """
    graph_builder = StateGraph(HumanProxyState)

    # ── Nodes ──
    graph_builder.add_node("evaluator", evaluator_node)
    graph_builder.add_node("actual_human", actual_human_node)

    # ── Edges ──
    graph_builder.add_edge(START, "evaluator")

    # After evaluator: resolve -> END, escalate -> actual_human
    graph_builder.add_conditional_edges(
        "evaluator",
        after_evaluator_router,
        {"actual_human": "actual_human", "__end__": END},
    )

    # Terminal edge
    graph_builder.add_edge("actual_human", END)

    agent_graph = graph_builder.compile()
    logger.info("Human Proxy agent graph compiled successfully.")
    return agent_graph
