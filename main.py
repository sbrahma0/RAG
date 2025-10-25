from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from groq_setup import get_groq_model

# for memory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

model = get_groq_model()

template = """
You are expert in answering questions about a pizza restaurant

Here are some relevant reviews: 
{reviews}

Here is the question to answer: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Wrapping the chain with memory to keep track of the conversation
session_store = {}


def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    
    return session_store[session_id]


runnable_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="reviews"
)

while True:
    print("\n\n --------------------------------------------------")
    question = input("Please enter your question (q to quit) - ")
    print("\n\n")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = runnable_with_memory.invoke({"question": question, "reviews": reviews},
                                         config={"configurable":{"session_id":"default"}})
    print(result.content if hasattr(result, "content") else result)
    print("-" * 80)

