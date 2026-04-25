from fastapi import APIRouter, UploadFile, File, Depends
from services.rag_service import upload_document, query_rag
from core.security import validate_token

router = APIRouter()

@router.post("/upload", dependencies=[Depends(validate_token)])
async def upload(file: UploadFile = File(...)):
    return await upload_document(file)


@router.post("/query", dependencies=[Depends(validate_token)])
def query(question: str):
    return query_rag(question)