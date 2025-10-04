from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
from io import StringIO
import joblib

from ..models.models import train_model  # Your training function


router = APIRouter(
    prefix="/train",
    tags=["Training"]
)

@router.post("/retrain/")
async def retrain(
    filename: str = Form(...),
    models: list[str] = Form(...),   # e.g., ["RandomForest", "AdaBoost"]
    learning_rate: float = Form(None),
    n_estimators: int = Form(None),
    max_depth: int = Form(None)
):
    try:
        # Load processed CSV
        PROCESSED_DIR = Path("storage/processed_data")
        train_file = PROCESSED_DIR / f"train_{filename}"
        test_file = PROCESSED_DIR / f"test_{filename}"
        if not train_file.exists() or not test_file.exists():
            raise HTTPException(status_code=404, detail="Processed CSV not found")
        
        # Merge with master dataset
        MASTER_DIR = Path("storage/master_data")
        MASTER_DIR.mkdir(parents=True, exist_ok=True)
        master_train_file = MASTER_DIR / "master_train.csv"
        master_test_file = MASTER_DIR / "master_test.csv"
        
        new_train = pd.read_csv(train_file)
        new_test = pd.read_csv(test_file)
        
        if master_train_file.exists():
            master_train = pd.read_csv(master_train_file)
            master_train = pd.concat([master_train, new_train], ignore_index=True)
        else:
            master_train = new_train
        
        if master_test_file.exists():
            master_test = pd.read_csv(master_test_file)
            master_test = pd.concat([master_test, new_test], ignore_index=True)
        else:
            master_test = new_test
        
        # Save updated master
        master_train.to_csv(master_train_file, index=False)
        master_test.to_csv(master_test_file, index=False)
        print(f"[INFO] Merged master train rows: {len(master_train)}")
        print(f"[INFO] Merged master test rows: {len(master_test)}")
        
        # Prepare features/target
        X_train, y_train = master_train.iloc[:, :-1], master_train.iloc[:, -1]
        X_test, y_test = master_test.iloc[:, :-1], master_test.iloc[:, -1]
        
        # Train selected models
        results = {}
        for model_name in models:
            model, metrics = train_model(
                X_train, y_train, X_test, y_test,
                model_name=model_name,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            # Save model
            MODEL_DIR = Path("app/services/ml/models")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_file = MODEL_DIR / f"{model_name}_latest.pkl"
            joblib.dump(model, model_file)
            results[model_name] = metrics
        
        return {"message": "Retraining complete", "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
