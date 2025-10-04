from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File
import pandas as pd
from ..service.ml.inference import inference

router = APIRouter(prefix='/ml', tags=['ml'])
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported", "status": "400"}

    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))
    inference
    
@router.get("/")
def hello_data():
    inference()
    return {"message": "Hello from FastAPI!"}



