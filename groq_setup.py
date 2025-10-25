import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_groq_model(model="llama-3.1-8b-instant"):
    """Initialize the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    llm = ChatGroq(
        model=model,  # or smaller if preferred
        api_key=api_key,
        temperature=0,
    )
    return llm
