import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

security = HTTPBearer()
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")