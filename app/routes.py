from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def home():
    return {"message": "Hello from FastAPI!"}

@router.get("/ping")
def ping():
    return {"status": "ok"}
