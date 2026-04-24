# Decoupled RAG Document Q&A API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI. This project demonstrates a decoupled AI architecture designed for optimal cost and latency.

## Architecture Highlights
*   **API Framework:** FastAPI for high-performance, asynchronous endpoints.
*   **Local Ingestion (Cost Optimization):** Document text extraction, chunking, and vector embeddings are performed locally using Hugging Face's `all-MiniLM-L6-v2` via `sentence-transformers`. This avoids heavy cloud API rate limits and costs during document processing.
*   **Vector Storage:** ChromaDB (local persistence).
*   **Cloud Inference (Latency Optimization):** Query generation is decoupled and routed to Groq's Language Processing Units (LPUs). Using `llama-3.1-8b-instant`, the API achieves near-instantaneous inference speeds for end-users.
*   **Orchestration:** LangChain.

## Endpoints

1.  `POST /upload`: Accepts a PDF/TXT file, chunks the text, generates embeddings locally, and stores them in ChromaDB.
2.  `POST /query`: Accepts a user query, retrieves the top-K relevant chunks from ChromaDB, and generates an answer using Groq's LLaMA 3.1 model based strictly on the retrieved context.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag-qa-api
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate # On Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

5. **Run the Server:**
   ```bash
   python -m uvicorn main:app --reload
   ```

6. **Test the API:**
   Navigate to `http://localhost:8000/docs` to interact with the Swagger UI.
