from src.tools.hitl import ask_human
from src.tools.file_ops import write_file, read_file, list_directory
from src.agents.blog_maker.src.tools.flowchart_maker import create_mermaid_flowchart


def get_tools():
    """Return all tools available to the blog_maker agent.

    Combines common tools (file ops, human-in-the-loop) with niche
    tools specific to blog creation (flowchart maker).
    """
    return [
        # Common tools
        ask_human,
        write_file,
        read_file,
        list_directory,
        # Niche tools
        create_mermaid_flowchart,
    ]
