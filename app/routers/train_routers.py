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
    file_train: str = Form(...),  # e.g., "train_data1.csv,train_data2.csv"
    file_test: str = Form(...),   # e.g., "test_data.csv"
    models: list[str] = Form(...),           # e.g., ["RandomForest", "AdaBoost"]
    learning_rate: float = Form(None),
    n_estimators: int = Form(None),
    max_depth: int = Form(None),
):
    try:
        PROCESSED_DIR = Path("storage/processed_data")

        # Prepare train DataFrame
        train_files = [f.strip() for f in file_train.split(",")]
        df_train = pd.concat([pd.read_csv(PROCESSED_DIR / f) for f in train_files], ignore_index=True)

        # Prepare test DataFrame
        test_files = [f.strip() for f in file_test.split(",")]
        df_test = pd.concat([pd.read_csv(PROCESSED_DIR / f) for f in test_files], ignore_index=True)

        # Validate
        if df_train.shape[1] < 2 or df_test.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least one feature column and one target column")

        # Split features/target
        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

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
