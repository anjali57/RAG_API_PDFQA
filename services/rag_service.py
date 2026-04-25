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


def query_rag(question: str):

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    result = query_document(question)

    return {"question": question, "answer": result["answer"]}