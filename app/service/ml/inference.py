import pandas as pd
from io import BytesIO
import joblib
import csv
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import os
from ..data.data_manage import merge_selected_csvs_to_inference

def load_model(model_type: int):
    """
    Load a machine learning model based on the selected model_type.
    model_type: int
        0 - AdaBoost
        1 - RandomForest
        2 - GradientBoosting
        3 - XGBoost
        4 - LightGBM
    """
    model_paths = {
        0: "app/service/ml/weights/adaboost_model.pkl",
        1: "app/service/ml/weights/random_forest_model.pkl",
        2: "app/service/ml/weights/extratree_model.pkl",
        3: "app/service/ml/weights/bagging_model.pkl",
        4: "app/service/ml/weights/stacking_model.pkl"
    }

    if model_type not in model_paths:
        raise ValueError(f"❌ Invalid model_type: {model_type}. Must be between 0–4.")

    model_path = model_paths[model_type]
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    return model

def inference_list_csvs(model_type, model_name, list_csv_names):
    """
    Run model inference and full evaluation on test data.
    Supports multiple model types.
    """
    model_path = os.path.join("app/storage/weights", model_type , model_name)
    model = joblib.load(model_path)
    

    final_data = merge_selected_csvs_to_inference(list_csv_names)
    # 4️⃣ Extract true labels
    if "koi_disposition" not in final_data.columns:
        raise ValueError("❌ 'koi_disposition' column not found in test CSV!")

    y_true = final_data["koi_disposition"]

    # 5️⃣ Drop label column from features
    # X_train = train_df.drop(columns=["koi_disposition"], errors="ignore")
    
    X_test = final_data.drop(columns=["koi_disposition"])

    feature_columns = [col for col in final_data.columns if col != "koi_disposition"]
    X_test = X_test[feature_columns]

    # ✅ Define scaler path
    just_model_name = Path(model_name).stem
    scaler_path = os.path.join("app/storage/scaler", str(model_type), f"{just_model_name}_scaler.pkl")

    # ✅ Load or fit the scaler
    if os.path.exists(scaler_path):
        print(f"🔹 Loading existing scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("⚠️ Scaler not found — fitting a new one on test data (not recommended for production).")
        scaler = StandardScaler()
        scaler.fit(X_test)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ New scaler saved to {scaler_path}")

    # ✅ Transform test data
    X_test = scaler.transform(X_test)
    print("✅ Test data scaled successfully!")


    # 7️⃣ Predict
    y_pred = model.predict(X_test.values)
    print("✅ Predictions completed!")

    # 8️⃣ Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 9️⃣ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=sorted(set(y_true)), columns=sorted(set(y_true)))

    # 🔟 Classification Report
    class_report = classification_report(y_true, y_pred, output_dict=True)

    # 11️⃣ Print readable results
    print("\n🎯 Model Evaluation Summary")
    print("=" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1-score (Macro): {f1_macro:.4f}")
    print("\nConfusion Matrix:")
    print(cm_df)
    print("\nDetailed Classification Report:")
    print(pd.DataFrame(class_report).T)

    return {
        "model_type": model_type,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm_df.to_dict(),
        "classification_report": class_report,
        "num_predictions": len(y_pred),
        "sample_predictions": list(map(str, y_pred[:10])),
    }






def inference_new_data(model_type, model_name, new_csvs_data):
    """
    Run model inference and full evaluation on test data.
    Supports multiple model types.
    """
    model_path = os.path.join("app/storage/weights", model_type , model_name, ".pkl")
    model = joblib.load(model_path)
    
    final_data = merge_selected_csvs(new_csvs_data)

    # 4️⃣ Extract true labels
    if "koi_disposition" not in final_data.columns:
        raise ValueError("❌ 'koi_disposition' column not found in test CSV!")

    y_true = final_data["koi_disposition"]

    # 5️⃣ Drop label column from features
    # X_train = train_df.drop(columns=["koi_disposition"], errors="ignore")
    X_test = final_data.drop(columns=["koi_disposition"], errors="ignore")

    X_test = X_test[final_data.columns]

    # 7️⃣ Predict
    y_pred = model.predict(X_test.values)
    print("✅ Predictions completed!")
    return y_pred


def inference_with_data(model_type, model_name, final_data):
    """
    Run model inference and full evaluation on test data.
    Supports multiple model types.
    """
    model_path = os.path.join("app/storage/weights", model_type , model_name, ".pkl")
    model = joblib.load(model_path)
    

    # 4️⃣ Extract true labels
    if "koi_disposition" not in final_data.columns:
        raise ValueError("❌ 'koi_disposition' column not found in test CSV!")

    y_true = final_data["koi_disposition"]

    # 5️⃣ Drop label column from features
    # X_train = train_df.drop(columns=["koi_disposition"], errors="ignore")
    X_test = final_data.drop(columns=["koi_disposition"], errors="ignore")

    X_test = X_test[final_data.columns]


    # 7️⃣ Predict
    y_pred = model.predict(X_test.values)
    print("✅ Predictions completed!")

    # 8️⃣ Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 9️⃣ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=sorted(set(y_true)), columns=sorted(set(y_true)))

    # 🔟 Classification Report
    class_report = classification_report(y_true, y_pred, output_dict=True)

    # 11️⃣ Print readable results
    print("\n🎯 Model Evaluation Summary")
    print("=" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1-score (Macro): {f1_macro:.4f}")
    print("\nConfusion Matrix:")
    print(cm_df)
    print("\nDetailed Classification Report:")
    print(pd.DataFrame(class_report).T)

    return {
        "model_type": model_type,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm_df.to_dict(),
        "classification_report": class_report,
        "num_predictions": len(y_pred),
        "sample_predictions": list(map(str, y_pred[:10])),
    }
