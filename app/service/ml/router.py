from fastapi import APIRouter
from pydantic import BaseModel
from service.ml.inference import predict_sentiment
<<<<<<< HEAD
<<<<<<< HEAD

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/predict")
def predict(input: TextInput):
    return predict_sentiment(input.text)
=======
=======
>>>>>>> 2a6e91e8b2023c193a6f1704267a5d81b3f01ac8
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

<<<<<<< HEAD
>>>>>>> 2a6e91e8b2023c193a6f1704267a5d81b3f01ac8
=======
>>>>>>> 2a6e91e8b2023c193a6f1704267a5d81b3f01ac8
