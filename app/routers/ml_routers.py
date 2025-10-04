from fastapi import APIRouter, Query
from fastapi import APIRouter, UploadFile, File
import pandas as pd
from ..service.ml.inference import inference_list_csvs, inference_new_data
from typing import List



router = APIRouter(prefix='/ml', tags=['ml'])
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported", "status": "400"}

    contents = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(contents))
    
    
@router.post("/inference/")
def inference_endpoint(
    model_type: str = Query(..., description="Model Type"),
    model_name: str = Query(..., description="Model Name"),
    list_csv_names: List[str] = Query(..., description="List of paths to testing CSV files")
):
    """
    Run model inference and analysis using provided CSV paths.
    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    data = inference_list_csvs(
        model_type=model_type,
        model_name=model_name,
        list_csv_names=list_csv_names  # updated parameter
    )
    return data


@router.post("/inference_new_csv_files/")
def inference_endpoint(
    model_type: str = Query(..., description="Model Type"),
    model_name: str = Query(..., description="Model Name"),
    list_csv_new: List[str] = Query(..., description="List of paths to testing CSV files")
):
    """
    Run model inference and analysis using provided CSV paths.
    Returns accuracy, precision, recall, F1, and confusion matrix.
    """
    data = inference_new_data(
        model_type=model_type,
        model_name=model_name,
        list_csv_names=list_csv_new  # updated parameter
    )
    return data
