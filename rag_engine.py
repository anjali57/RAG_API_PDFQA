import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load the API key from the .env file
load_dotenv()

# Disable ChromaDB Telemetry to fix the warning
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# We need a place to save our Vector Database on the hard drive
CHROMA_DB_DIRECTORY = "./chroma_db"

# Initialize the tools
# Embeddings map text to numbers. We use a lightweight open-source model that runs locally on your CPU.
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# The LLM answers the questions. Groq uses LPUs to run open-source models incredibly fast.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=512
)

# Initialize Chroma Vector Store (if it doesn't exist, it will be created)
vector_store = Chroma(
    embedding_function=embeddings_model, 
    persist_directory=CHROMA_DB_DIRECTORY
)

def process_document_to_chroma(file_bytes: bytes, filename: str):
    """
    Takes a PDF file as raw bytes, extracts the text, splits it into smaller chunks,
    generates embeddings for each chunk, and saves them to ChromaDB.
    """
    # 1. We have to save the bytes to a temporary file because PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        # 2. Extract text from the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 3. Split the text into manageable chunks
        # We don't want chunks too big, otherwise we exceed the LLM's memory
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200 # Overlap helps keep context between chunks
        )
        chunks = text_splitter.split_documents(documents)

        # 4. Generate embeddings and store them in ChromaDB
        # This will run locally on your computer and might take a few seconds per page
        if chunks:
            vector_store.add_documents(chunks)
            
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def query_rag_system(question: str) -> str:
    """
    Takes a user's question, searches ChromaDB for relevant chunks,
    and asks the LLM to answer the question based only on those chunks.
    """
    # 1. Setup the Retriever (the search engine for our Vector Database)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 most relevant chunks

    # 2. Define the Prompt (Instructions for the LLM)
    # Mistral prefers a specific prompt format
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

    # 3. Create the Chain (Glue everything together)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 4. Execute the chain and get the answer
    response = rag_chain.invoke({"input": question})
    
    # Return the text of the answer
    return response["answer"]
