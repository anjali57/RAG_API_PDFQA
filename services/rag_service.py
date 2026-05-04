from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from repositories.rag_repository import process_document, query_document

async def upload_document(file: UploadFile):
    
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported")

    contents = await file.read()

    process_document(contents, file.filename)

    return JSONResponse(
        content={"message": f"{file.filename} processed successfully"}
    )

def query_rag(question: str, session_id: str):

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    result = query_document(question, session_id)
    
    sources = []
    # The 'context' key contains the list of Document objects retrieved from ChromaDB
    if "context" in result:
        for doc in result["context"]:
            # Extract metadata (like source file and page) and a snippet of the text
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)

    return {
        "question": question, 
        "answer": result["answer"],
        "sources": sources
    }