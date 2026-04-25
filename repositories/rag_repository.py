import os
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from db.vector_store import vector_store

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)

def process_document(file_bytes: bytes, filename: str):
    """
    Takes a file as raw bytes, extracts the text, splits it into smaller chunks,
    generates embeddings for each chunk, and saves them to ChromaDB.
    """
    suffix = os.path.splitext(filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif suffix == ".txt":
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        if chunks:
            vector_store.add_documents(chunks)
            
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def query_document(question: str) -> dict:
    """
    Takes a user's question, searches ChromaDB for relevant chunks,
    and asks the LLM to answer the question based only on those chunks.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    
    return response