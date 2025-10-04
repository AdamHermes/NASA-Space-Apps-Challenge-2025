from fastapi import APIRouter
from pydantic import BaseModel
from service.ml.inference import predict_sentiment

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/predict")
def predict(input: TextInput):
    return predict_sentiment(input.text)

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
