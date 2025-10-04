from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
from fastapi.responses import JSONResponse
import os
from typing import List
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
async def process_csv(filenames: List[str], option: str = Form(...)):
    try:
        filename_list = [f.strip() for f in filenames[0].split(',')]
        # filename_list = filenames
        print(filename_list)
        result = process_koi(filename_list)
        return {
            "message": f"CSV processed with option '{option}'",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current_models")
def get_models():
    return get_current_models()

@router.get("/current_csvs")
def get_data():
    return get_current_csv_files()
