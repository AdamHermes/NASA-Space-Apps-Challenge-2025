from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
from fastapi.responses import JSONResponse
import os
import shutil
import pandas as pd 
from ..data_services.process_koi import process_koi


router = APIRouter(prefix='/data', tags=['data'])

CSV_UPLOAD_DIR = Path("storage/uploaded_csvs")

@router.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    file_path = CSV_UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "filename": file.filename}

@router.post("/process_csv/")
async def process_csv(
    filename: str = Form(...),
    option: str = Form(...)
):
    try:
        result = process_koi(filename)
        return {
            "message": f"CSV processed with option '{option}'",
            "train_file": result["train_file"],
            "test_file": result["test_file"]
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    