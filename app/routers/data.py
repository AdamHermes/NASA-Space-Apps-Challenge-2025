from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
from fastapi.responses import JSONResponse
import os
import shutil
import pandas as pd 
from ..service.data.process_koi import process_koi
from ..service.data.data_manage import get_current_models, get_current_csv_files


router = APIRouter(prefix='/data', tags=['data'])

CSV_UPLOAD_DIR = Path("app/storage/uploaded_csvs")

@router.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    file_path = CSV_UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = pd.read_csv(file_path, comment="#")
    df = df.fillna("nan")

    return {"message": "File uploaded successfully", "filename":file.filename, "filepath": file_path, "data_head": df.head().to_dict(orient="records")}

@router.post("/process_csv/")
async def process_csv(filename: str = Form(...), option: str = Form(...)):
    try:
        result = process_koi(filename)
        # return {
        #     "message": f"CSV processed with option '{option}'",
        #     "train": result["train_file"],
        #     "test_file": result["test_file"]
        # }
        return {
            "message": f"CSV processed with option '{option}'",
            "train_filename": result["train_filename"],
            "train_filepath": result["train_filepath"],
            "test_filename": result["test_filename"],
            "test_filepath": result["test_filepath"],
            "train_head": result["train_head"],
            "test_head": result["test_head"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@router.get("/current_models")
def get_models():
    return get_current_models()

@router.get("/current_csvs")
def get_data():
    return get_current_csv_files()