from fastapi import APIRouter
import os
from fastapi.responses import JSONResponse
import shutil

router = APIRouter()

@router.get("/")
def home():
    return {"message": "Hello from FastAPI!"}

@router.get("/ping")
def ping():
    return {"status": "ok"}

