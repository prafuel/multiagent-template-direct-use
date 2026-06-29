from langchain_core.tools import tool

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question when you need clarification, additional
    context, or permission before proceeding. Use this when:
    - The user's query is ambiguous or underspecified
    - You need permission to perform an action (e.g., searching the internet)
    - You need additional details to give a complete answer
    - You hit a dead end and need guidance on how to proceed

    Args:
        question: The question to ask the human user.
    """
    print(f"\n{'-'*40}")
    print("HUMAN-IN-THE-LOOP (Agent is asking you a question)")
    print(f"{'-'*40}")
    print(f"QUE : {question}")
    print(f"{'-'*40}")
    user_response = input("Your response: ").strip()
    return f"Feedback: {user_response}"
