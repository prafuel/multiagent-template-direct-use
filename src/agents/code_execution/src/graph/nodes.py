"""
Nodes and routing logic for the Code Execution Agent.

Graph flow:
  START -> coder -> approval_agent -> [auto_approved]  -> execute_code -> END
                                   -> [escalate]       -> human_proxy  -> [human_approved] -> execute_code -> END
                                                                       -> [retry]          -> coder (loop back)
                                                                       -> [aborted]        -> abort -> END
"""

import os
import re
import time
import logging
import subprocess

from typing import Literal

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from src.config import get_llm
from src.agents.code_execution.src.graph.states import CodeExecutionState
from src.agents.code_execution.src.graph.prompts import (
    CODER_SYSTEM_PROMPT,
    APPROVAL_SYSTEM_PROMPT,
)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MAX_EXECUTION_TIME = int(os.getenv("MAX_EXECUTION_TIME_SECONDS", "120"))

logger = logging.getLogger("agent.code_execution")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> str:
    """Extract a Python code block from fenced markdown.

    Looks for ```python ... ``` first, then falls back to ``` ... ```.
    If no fenced block is found, returns the raw text stripped.
    """
    # Try ```python ... ```
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fenced block — return raw text
    return text.strip()


def _parse_approval_decision(text: str) -> str:
    """Parse the approval agent's output into a status string.

    Expected format:
        DECISION: auto_approved
        REASON: ...
    """
    text_lower = text.lower()
    if "auto_approved" in text_lower:
        return "auto_approved"
    if "escalate" in text_lower:
        return "escalate"
    # Default to escalate if we can't parse — fail-safe
    logger.warning("[APPROVAL] Could not parse decision from: %s. Defaulting to 'escalate'.", text[:200])
    return "escalate"


# ─────────────────────────────────────────────────────────────────────────────
#  NODES
# ─────────────────────────────────────────────────────────────────────────────

def coder_node(state: CodeExecutionState) -> dict:
    """Generate a Python code snippet for the user's task."""
    iteration = state.get("iteration", 0) + 1
    elapsed = time.time() - state.get("start_time", time.time())

    logger.info("\n" + "=" * 60)
    logger.info("[CODER]  Iteration %d/%d  |  Elapsed: %.1fs", iteration, MAX_ITERATIONS, elapsed)
    logger.info("=" * 60)

    # ── Safeguards ──
    if elapsed > MAX_EXECUTION_TIME:
        logger.warning("[CODER]  TIMEOUT. Forcing end.")
        return {
            "messages": [AIMessage(content="Exceeded the time limit while generating code.")],
            "iteration": iteration,
            "approval_status": "aborted",
        }

    if iteration > MAX_ITERATIONS:
        logger.warning("[CODER]  MAX ITERATIONS reached. Forcing end.")
        return {
            "messages": [AIMessage(content="Reached maximum iteration count during code generation.")],
            "iteration": iteration,
            "approval_status": "aborted",
        }

    # Build messages for the LLM
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

    # If this is a retry, inject the feedback
    retry_feedback = state.get("retry_feedback", "")
    if retry_feedback:
        non_system.append(HumanMessage(content=f"Please revise the code based on this feedback: {retry_feedback}"))

    messages = [CODER_SYSTEM_PROMPT] + non_system

    llm = get_llm()
    response = llm.invoke(messages)
    code = _extract_code_block(response.content)

    logger.info("[CODER]  Generated code (%d chars):", len(code))
    logger.info("-" * 40)
    for line in code.split("\n"):
        logger.info("  %s", line)
    logger.info("-" * 40)

    return {
        "messages": [response],
        "iteration": iteration,
        "code_snippet": code,
        "retry_feedback": "",  # clear any previous feedback
    }


def approval_agent_node(state: CodeExecutionState) -> dict:
    """LLM-based safety reviewer. Decides auto_approved or escalate."""
    code = state.get("code_snippet", "")
    task_description = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            task_description = m.content
            break

    logger.info("\n" + "=" * 60)
    logger.info("[APPROVAL AGENT]  Reviewing code...")
    logger.info("=" * 60)

    review_prompt = HumanMessage(content=f"""\
TASK: {task_description}

CODE:
```python
{code}
```

Please classify this code as auto_approved or escalate.""")

    llm = get_llm()
    response = llm.invoke([APPROVAL_SYSTEM_PROMPT, review_prompt])
    decision = _parse_approval_decision(response.content)

    logger.info("[APPROVAL AGENT]  Decision: %s", decision.upper())
    logger.info("[APPROVAL AGENT]  Reasoning: %s", response.content.strip())

    return {
        "approval_status": decision,
    }


def human_proxy_node(state: CodeExecutionState) -> dict:
    """Synchronous human-in-the-loop node. Only reached when approval_agent escalates.

    Displays the code and asks the human to:
      - Type 'approve' to proceed with execution
      - Type 'abort' to cancel
      - Type anything else to treat as retry feedback
    """
    code = state.get("code_snippet", "")

    print("\n" + "=" * 60)
    print("  HUMAN REVIEW REQUIRED — Code Execution Agent")
    print("=" * 60)
    print("\nThe following code has been flagged for human review:\n")
    print("-" * 40)
    print(code)
    print("-" * 40)
    print("\nOptions:")
    print("  approve  — Execute this code")
    print("  abort    — Cancel execution")
    print("  <other>  — Provide feedback to regenerate the code")
    print()

    user_input = input("Your decision: ").strip()

    if not user_input:
        user_input = "abort"

    choice = user_input.lower()

    if choice == "approve":
        logger.info("[HUMAN PROXY]  Human APPROVED execution.")
        return {
            "approval_status": "human_approved",
        }
    elif choice == "abort":
        logger.info("[HUMAN PROXY]  Human ABORTED execution.")
        return {
            "approval_status": "aborted",
        }
    else:
        logger.info("[HUMAN PROXY]  Human requested RETRY with feedback: %s", user_input)
        return {
            "approval_status": "retry",
            "retry_feedback": user_input,
        }


def execute_code_node(state: CodeExecutionState) -> dict:
    """Execute the approved Python code locally using subprocess."""
    code = state.get("code_snippet", "")

    logger.info("\n" + "=" * 60)
    logger.info("[EXECUTOR]  Running approved code...")
    logger.info("=" * 60)

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        exit_code = result.returncode

        logger.info("[EXECUTOR]  Exit code: %d", exit_code)
        if stdout:
            logger.info("[EXECUTOR]  STDOUT:\n%s", stdout)
        if stderr:
            logger.info("[EXECUTOR]  STDERR:\n%s", stderr)

        if exit_code == 0:
            output_msg = f"Code executed successfully (exit code 0).\n\nOutput:\n{stdout}" if stdout else "Code executed successfully (exit code 0). No output."
        else:
            output_msg = f"Code failed (exit code {exit_code}).\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

    except subprocess.TimeoutExpired:
        output_msg = "Code execution timed out (30s limit)."
        logger.error("[EXECUTOR]  TIMEOUT")
    except Exception as e:
        output_msg = f"Code execution error: {type(e).__name__}: {e}"
        logger.error("[EXECUTOR]  ERROR: %s", output_msg)

    return {
        "messages": [AIMessage(content=output_msg)],
    }


def abort_node(state: CodeExecutionState) -> dict:
    """Handle the abort path gracefully."""
    logger.info("\n" + "=" * 60)
    logger.info("[ABORT]  Code execution was cancelled.")
    logger.info("=" * 60)

    return {
        "messages": [AIMessage(content="Code execution has been cancelled. No code was run.")],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def after_approval_router(state: CodeExecutionState) -> Literal["execute_code", "human_proxy"]:
    """Route after the approval agent's decision.

    auto_approved -> execute_code
    escalate      -> human_proxy
    """
    status = state.get("approval_status", "escalate")

    if status == "auto_approved":
        logger.info("[ROUTER]  Approval: auto_approved -> execute_code")
        return "execute_code"
    else:
        logger.info("[ROUTER]  Approval: escalate -> human_proxy")
        return "human_proxy"


def after_human_proxy_router(state: CodeExecutionState) -> Literal["execute_code", "abort", "coder"]:
    """Route after the human proxy's decision.

    human_approved -> execute_code
    aborted        -> abort
    retry          -> coder (loop back with feedback)
    """
    status = state.get("approval_status", "aborted")

    if status == "human_approved":
        logger.info("[ROUTER]  Human: approved -> execute_code")
        return "execute_code"
    elif status == "retry":
        logger.info("[ROUTER]  Human: retry -> coder")
        return "coder"
    else:
        logger.info("[ROUTER]  Human: aborted -> abort")
        return "abort"


def after_coder_router(state: CodeExecutionState) -> Literal["approval_agent", "__end__"]:
    """Route after the coder node. If the coder aborted (safeguards), go to END.
    Otherwise proceed to approval."""
    status = state.get("approval_status", "")

    if status == "aborted":
        logger.info("[ROUTER]  Coder safeguard triggered -> END")
        return "__end__"

    logger.info("[ROUTER]  Coder -> approval_agent")
    return "approval_agent"
