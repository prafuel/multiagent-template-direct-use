from src.get_tools import get_tools as get_all_common_tools
from src.agents.planning.src.tools.wikipedia_search import wiki_tool

def get_tools():
    tools = get_all_common_tools() + [wiki_tool]
    return tools