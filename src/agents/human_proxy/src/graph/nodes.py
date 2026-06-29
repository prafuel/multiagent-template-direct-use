"""
Nodes and routing logic for the Human Proxy Agent.

Graph flow:
  START -> evaluator -> [resolve]   -> END  (proxy answered autonomously)
                     -> [escalate]  -> actual_human -> END
"""

import json
import logging

from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage

from src.config import get_llm
from src.agents.human_proxy.src.graph.states import HumanProxyState
from src.agents.human_proxy.src.graph.prompts import PROXY_EVALUATOR_SYSTEM_PROMPT

logger = logging.getLogger("agent.human_proxy")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_query(state: HumanProxyState) -> str:
    """Extract the query/question from the message history.

    Looks for the last HumanMessage in the conversation, which should contain
    the question or decision request from the calling agent/orchestrator.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _parse_evaluator_response(text: str) -> dict:
    """Parse the evaluator LLM's JSON response.

    Returns a dict with keys: decision, confidence, response, reasoning.
    Falls back to escalation if parsing fails.
    """
    # Try to find JSON in the response (handle markdown fencing)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip markdown code fences
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        # Validate required keys
        decision = parsed.get("decision", "escalate")
        confidence = parsed.get("confidence", "low")
        response = parsed.get("response", "")
        reasoning = parsed.get("reasoning", "")

        # Safety: if confidence is not high, force escalation
        if confidence in ("low", "medium") and decision == "resolve":
            logger.warning(
                "[EVALUATOR] Overriding resolve->escalate due to %s confidence.",
                confidence,
            )
            decision = "escalate"
            reasoning = f"Confidence too low ({confidence}) to resolve autonomously. {reasoning}"

        return {
            "decision": decision,
            "confidence": confidence,
            "response": response,
            "reasoning": reasoning,
        }

    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(
            "[EVALUATOR] Failed to parse JSON response: %s. Defaulting to escalate.",
            e,
        )
        return {
            "decision": "escalate",
            "confidence": "low",
            "response": "",
            "reasoning": f"Could not parse evaluator response. Raw: {text[:200]}",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  NODES
# ─────────────────────────────────────────────────────────────────────────────

def evaluator_node(state: HumanProxyState) -> dict:
    """Evaluate whether the incoming query can be resolved by the proxy.

    Invokes the LLM with the proxy evaluator prompt and parses the decision.
    """
    query = _extract_query(state)

    logger.info("\n" + "=" * 60)
    logger.info("[HUMAN PROXY]  Evaluating query...")
    logger.info("  | Query: %s", query[:200])
    logger.info("=" * 60)

    if not query:
        logger.warning("[HUMAN PROXY]  No query found in messages. Escalating.")
        return {
            "query": "",
            "decision": "escalate",
            "confidence": "low",
            "escalation_reason": "No query found in message history.",
        }

    # Build the evaluation request
    eval_request = HumanMessage(content=f"""\
The following query has been sent to the human for a decision. Evaluate whether
you can answer it on behalf of the human, or whether it must be escalated.

QUERY:
{query}
""")

    llm = get_llm()
    response = llm.invoke([PROXY_EVALUATOR_SYSTEM_PROMPT, eval_request])

    parsed = _parse_evaluator_response(response.content)

    logger.info("[HUMAN PROXY]  Decision: %s", parsed["decision"].upper())
    logger.info("[HUMAN PROXY]  Confidence: %s", parsed["confidence"])
    logger.info("[HUMAN PROXY]  Reasoning: %s", parsed["reasoning"])

    if parsed["decision"] == "resolve":
        logger.info("[HUMAN PROXY]  Proxy response: %s", parsed["response"])
        return {
            "query": query,
            "decision": "resolve",
            "confidence": parsed["confidence"],
            "proxy_response": parsed["response"],
            "escalation_reason": "",
            "messages": [
                AIMessage(
                    content=(
                        f"[Human Proxy - Auto-resolved] {parsed['response']}"
                    )
                )
            ],
        }
    else:
        return {
            "query": query,
            "decision": "escalate",
            "confidence": parsed["confidence"],
            "proxy_response": "",
            "escalation_reason": parsed["reasoning"],
        }


def actual_human_node(state: HumanProxyState) -> dict:
    """Synchronous fallback: ask the real human for input.

    This node is only reached when the evaluator decides the query
    genuinely requires human attention.
    """
    query = state.get("query", "")
    reason = state.get("escalation_reason", "Unknown reason")

    print("\n" + "=" * 60)
    print("  HUMAN INPUT REQUIRED — Human Proxy Agent")
    print("=" * 60)
    print(f"\nThe AI proxy decided to escalate this to you.")
    print(f"Reason: {reason}")
    print(f"\nQuery requiring your attention:\n")
    print("-" * 40)
    print(query)
    print("-" * 40)
    print()

    user_input = input("Your response: ").strip()

    if not user_input:
        user_input = "No response provided by human."

    logger.info("[HUMAN PROXY]  Real human responded: %s", user_input[:200])

    return {
        "proxy_response": user_input,
        "messages": [
            AIMessage(
                content=f"[Human Proxy - Escalated to human] {user_input}"
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def after_evaluator_router(state: HumanProxyState) -> Literal["actual_human", "__end__"]:
    """Route after the evaluator's decision.

    resolve  -> END (proxy already answered in evaluator_node)
    escalate -> actual_human
    """
    decision = state.get("decision", "escalate")

    if decision == "resolve":
        logger.info("[ROUTER]  Proxy resolved -> END")
        return "__end__"
    else:
        logger.info("[ROUTER]  Escalating -> actual_human")
        return "actual_human"
