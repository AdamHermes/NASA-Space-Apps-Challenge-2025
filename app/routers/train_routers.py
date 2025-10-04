from fastapi import APIRouter, Form, HTTPException
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

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
    max_depth: int = Form(None),
    test_size: float = Form(0.2),            # Fraction of data for testing
    random_state: int = Form(42)             # For reproducibility
):
    try:
        # Load merged CSV
        MERGED_DIR = Path("storage/merged_csvs")
        csv_file = MERGED_DIR / filename
        if not csv_file.exists():
            raise HTTPException(status_code=404, detail="Merged CSV not found")
        
        df = pd.read_csv(csv_file)
        
        # Check that CSV has at least one feature and one target column
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least one feature column and one target column")
        
        # Split features/target
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
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
