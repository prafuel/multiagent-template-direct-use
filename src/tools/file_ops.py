import os
from langchain_core.tools import tool


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file on disk. Creates parent directories if they
    don't exist. Overwrites the file if it already exists.

    Args:
        file_path: The absolute or relative path of the file to write.
        content: The text content to write into the file.
    """
    try:
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {abs_path}"
    except Exception as e:
        return f"Error writing file: {type(e).__name__}: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file from disk.

    Args:
        file_path: The absolute or relative path of the file to read.
    """
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"File not found: {abs_path}"
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"File contents of {abs_path} ({len(content)} chars):\n\n{content}"
    except Exception as e:
        return f"Error reading file: {type(e).__name__}: {e}"


@tool
def list_directory(directory_path: str) -> str:
    """List the contents of a directory, showing files and subdirectories.

    Args:
        directory_path: The absolute or relative path of the directory to list.
    """
    try:
        abs_path = os.path.abspath(directory_path)
        if not os.path.isdir(abs_path):
            return f"Not a directory: {abs_path}"

        entries = os.listdir(abs_path)
        if not entries:
            return f"Directory {abs_path} is empty."

        lines = [f"Contents of {abs_path} ({len(entries)} items):"]
        for entry in sorted(entries):
            full = os.path.join(abs_path, entry)
            if os.path.isdir(full):
                lines.append(f"  [DIR] {entry}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"  {entry}  ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {type(e).__name__}: {e}"
