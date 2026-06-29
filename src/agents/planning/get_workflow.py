# ====================== below code is just for template ===============================

import os
import logging
from langgraph.graph import StateGraph, START, END

from src.agents.planning.src.graph.states import AgentState
from src.agents.planning.src.graph.nodes import ( 
    planner_node, 
    agent_node, 
    tool_executor_node, 
    after_agent_router
)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))  # keep last N message-pairs

logger = logging.getLogger("agent.planning")

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("planner", planner_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_executor_node)

# Edges: START -> planner -> agent <-> tools -> END
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "agent")
graph_builder.add_conditional_edges("agent", after_agent_router, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "agent")  # after tools, always go back to agent for next decision


def get_workflow():
    # Compile
    agent_graph = graph_builder.compile()
    logger.info("Planning agent graph compiled successfully (MAX_ITERATIONS=%d, MEMORY_WINDOW=%d)", MAX_ITERATIONS, MEMORY_WINDOW)
    return agent_graph