from fastapi import APIRouter
from pydantic import BaseModel
from service.ml.inference import predict_sentiment
from fastapi import APIRouter, UploadFile, File
import pandas as pd

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported", "status": "400"}

    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))

    pass

