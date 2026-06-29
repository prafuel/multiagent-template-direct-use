from src.tools.hitl import ask_human
from src.tools.internet_search import search_tool
from src.tools.file_ops import write_file, read_file, list_directory

def get_tools():
    return [
        write_file, read_file, list_directory, ask_human, search_tool
    ]