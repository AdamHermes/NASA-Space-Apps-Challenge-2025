from fastapi import APIRouter
from service.ml import router as ai_router

router = APIRouter()
router.include_router(ai_router.router, prefix="/ml", tags=["Machine Learning Service"])