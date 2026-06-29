def get_agent_description() -> str:
    """Return a short description of this agent for the orchestrator."""
    return (
        "Blog Maker Agent — Writes highly optimized, well-structured blog posts "
        "using industry-standard templates (How-To, Listicle, Pillar, Case Study, "
        "Tutorial, Opinion, FAQ, etc.). Can also create Mermaid flowcharts to "
        "visualize blog structure. Has access to file system tools for saving "
        "drafts and reading reference material." + 

        "Save blog draft at following location ./output"
    )
