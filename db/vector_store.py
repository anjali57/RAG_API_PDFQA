import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_DIRECTORY = "./chroma_db"

embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    embedding_function=embeddings_model,
    persist_directory=CHROMA_DB_DIRECTORY
)