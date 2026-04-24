from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import List

from rag_engine import process_document_to_chroma, query_rag_system

app = FastAPI(
    title="RAG Document Q&A API",
    description="An API to upload documents and ask questions about them using RAG.",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Document Q&A API. Go to /docs for the interactive API documentation."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or Text) to be processed and added to the RAG system.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    try:
        contents = await file.read()
        # Call our logic from rag_engine.py
        process_document_to_chroma(contents, file.filename)
        
        return JSONResponse(content={"message": f"Successfully uploaded and processed {file.filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(question: str):
    """
    Ask a question about the uploaded documents.
    """
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # Call our logic from rag_engine.py
        answer = query_rag_system(question)
        
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
