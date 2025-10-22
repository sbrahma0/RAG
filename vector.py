from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_lanchain_db"
add_docs = not os.path.exists(db_location)

if add_docs:
    docs = []
    ids = []

    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(i)
        docs.append(doc)

vectorstore = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
    )

if add_docs:
    vectorstore.add_documents(documents=docs, ids=ids)
    vectorstore.persist()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
