from fastapi import APIRouter, Form, HTTPException
import pandas as pd
from pathlib import Path
import joblib

from ..models.models import train_model  # Your training function

router = APIRouter(
    prefix="/train",
    tags=["Training"]
)

@router.post("/retrain/")
async def retrain(
    filename: str = Form(...),                # Name of the merged CSV
    models: list[str] = Form(...),           # e.g., ["RandomForest", "AdaBoost"]
    learning_rate: float = Form(None),
    n_estimators: int = Form(None),
    max_depth: int = Form(None)
):
    try:
        # Load merged CSV
        MERGED_DIR = Path("storage/merged_csvs")
        train_file = MERGED_DIR / filename
        if not train_file.exists():
            raise HTTPException(status_code=404, detail="Merged CSV not found")
        
        df = pd.read_csv(train_file)
        
        # Split features/target
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least one feature column and one target column")
        
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        
        # Train selected models
        results = {}
        for model_name in models:
            model, metrics = train_model(
                X, y, X, y,                  # Using same data as train/test for simplicity, can adjust if needed
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
