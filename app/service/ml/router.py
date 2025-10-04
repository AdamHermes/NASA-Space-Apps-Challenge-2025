from fastapi import APIRouter
from pydantic import BaseModel
from service.ml.inference import predict_sentiment

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/predict")
def predict(input: TextInput):
    return predict_sentiment(input.text)
