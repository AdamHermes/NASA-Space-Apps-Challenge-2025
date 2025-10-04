from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
from fastapi.responses import JSONResponse
import os
import shutil
import pandas as pd 
from ..service.data.process_koi import process_koi


router = APIRouter(prefix='/data', tags=['data'])

CSV_UPLOAD_DIR = Path("storage/uploaded_csvs")

@router.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    file_path = CSV_UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "filename": file.filename}

@router.post("/process_csv/")
async def process_csv(filename: str = Form(...), option: str = Form(...)):
    try:
        CSV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CSV_UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        result = process_koi(file_path)  # returns {"train_file": df, "test_file": df}
        
        # Save processed CSV temporarily
        PROCESSED_DIR = Path("storage/processed_data")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_file = PROCESSED_DIR / f"train_{filename}"
        test_file = PROCESSED_DIR / f"test_{filename}"
        result["train_file"].to_csv(train_file, index=False)
        result["test_file"].to_csv(test_file, index=False)
        
        return {
            "message": f"CSV processed with option '{option}'",
            "train_file": str(train_file),
            "test_file": str(test_file)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
