import os
import logging

from langgraph.graph import StateGraph, START, END

from src.agents.blog_maker.src.graph.states import BlogMakerState
from src.agents.blog_maker.src.graph.nodes import (
    blog_agent_node,
    blog_tool_executor_node,
    after_blog_agent_router,
)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

logger = logging.getLogger("agent.blog_maker")


def get_workflow():
    """Build and compile the blog_maker agent graph."""
    graph_builder = StateGraph(BlogMakerState)

    graph_builder.add_node("agent", blog_agent_node)
    graph_builder.add_node("tools", blog_tool_executor_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent", after_blog_agent_router, {"tools": "tools", "__end__": END}
    )
    graph_builder.add_edge("tools", "agent")

    agent_graph = graph_builder.compile()
    logger.info("Blog Maker agent graph compiled successfully (MAX_ITERATIONS=%d)", MAX_ITERATIONS)
    return agent_graph
