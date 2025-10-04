from fastapi import APIRouter, Query
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
    
@router.post("/inference/")
def inference_endpoint(
    model_type: int = Query(0, ge=0, le=4, description="0=Adaboost, 1=RandomForest, 2=GradientBoosting, 3=XGBoost, 4=LightGBM"),
    train_path: str = Query(..., description="Path to the training CSV file"),
    test_path: str = Query(..., description="Path to the testing CSV file")
):
    """
    Run model inference and analysis using provided CSV paths.
    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    data = inference(model_type=model_type, train_path=train_path, test_path=test_path)
    return data


