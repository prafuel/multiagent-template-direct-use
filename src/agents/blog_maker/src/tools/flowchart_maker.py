from langchain_core.tools import tool


@tool
def create_mermaid_flowchart(title: str, description: str, steps: list[str]) -> str:
    """Generate a Mermaid flowchart diagram based on the provided information
    and return it as a markdown-ready string. Use this when you need to
    visualize a blog post structure, workflow, or process as a flowchart.

    Args:
        title: The title for the flowchart.
        description: A brief description of what the flowchart represents.
        steps: A list of step descriptions to include in the flowchart.
               Each step becomes a node in the diagram.
    """
    if not steps:
        return "Error: At least one step is required to create a flowchart."

    # Build the Mermaid diagram
    lines = [
        f"# {title}",
        f"_{description}_",
        "",
        "```mermaid",
        "flowchart TD",
    ]

    # Create nodes with sanitized IDs
    for i, step in enumerate(steps):
        node_id = f"step{i}"
        # Sanitize step text for Mermaid (escape quotes)
        safe_text = step.replace('"', "'")
        lines.append(f'    {node_id}["{safe_text}"]')

    # Create edges between consecutive nodes
    lines.append("")
    for i in range(len(steps) - 1):
        lines.append(f"    step{i} --> step{i+1}")

    # Style the start and end nodes
    if len(steps) >= 1:
        lines.append("")
        lines.append(f"    style step0 fill:#4CAF50,color:#fff,stroke:#333")
    if len(steps) >= 2:
        last = len(steps) - 1
        lines.append(f"    style step{last} fill:#2196F3,color:#fff,stroke:#333")

    lines.append("```")

    mermaid_content = "\n".join(lines)
    return f"Mermaid flowchart generated:\n\n{mermaid_content}"
