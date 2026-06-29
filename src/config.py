import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# llama-3.3-70b-versatile
MODEL_NAME = os.getenv("MODEL_NAME")

def get_llm(**kwargs):
    """Get configured LLM instance"""
    LLM = ChatGroq(
        model=MODEL_NAME,
        # temperature=0,
        verbose=True,
        **kwargs
    )
    return LLM

# llm = get_llm()
# print(llm.invoke("hi"))