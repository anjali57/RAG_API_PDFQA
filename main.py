from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="RAG Document Q&A API",
    description="An API to upload documents and ask questions about them using RAG.",
    version="0.1.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "RAG API running"}
