import json
import os

from langchain_core.messages import SystemMessage
from src.agents.blog_maker.get_tools import get_tools

# ── Load blog template data ──────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

with open(os.path.join(_DATA_DIR, "blog_template_usage.json"), "r") as f:
    _template = json.load(f)
    BLOG_USAGE_GUIDELINES = _template["Template_Usage_Guidelines"]

with open(os.path.join(_DATA_DIR, "blog_templates_detailed.json"), "r") as f:
    BLOG_TEMPLATES = json.load(f)

# Build a concise summary of available template types
_TEMPLATE_NAMES = ", ".join(BLOG_TEMPLATES.keys())


def get_blog_maker_prompt() -> SystemMessage:
    """Build the full system prompt for the blog_maker agent."""
    tool_info = "\n".join(
        f"  - {tool.name}: {tool.description.split(chr(10))[0]}"
        for tool in get_tools()
    )

    content = f"""\
You are the Blog Maker Agent — an expert content writer specializing in creating
high-quality, SEO-optimized blog posts using proven templates.

═══════════════════════════════════════════════════════════════
AVAILABLE TOOLS:
{tool_info}
═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
BLOG TEMPLATE USAGE GUIDELINES:
{BLOG_USAGE_GUIDELINES}
═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
AVAILABLE BLOG TEMPLATES:
{_TEMPLATE_NAMES}
═══════════════════════════════════════════════════════════════

WORKFLOW:
1. Analyze the user's request to determine the best blog template.
2. If the request is unclear, use ask_human to clarify the topic, audience, 
   and desired template.
3. Select and follow the appropriate template structure from your knowledge.
4. Optionally create a flowchart of the blog structure using create_mermaid_flowchart.
5. Write the full blog post following the template guidelines.
6. Save the final blog post using write_file.

IMPORTANT RULES:
- Always follow the chosen template structure precisely.
- Ensure the blog meets the minimum word count requirement (800-1,000 words minimum).
- Include SEO-optimized titles (60 characters or less).
- Add a compelling call-to-action at the end.
- Use the write_file tool to save the final output.
- Add unique perspective and avoid generic AI-sounding content.
"""
    return SystemMessage(content=content)
