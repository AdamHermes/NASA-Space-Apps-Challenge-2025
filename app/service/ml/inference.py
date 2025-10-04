import pandas as pd
from io import BytesIO
import joblib
import csv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



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
        0: "app/service/ml/models/adaboost_model.pkl",
        1: "app/service/ml/models/random_forest_model.pkl",
        2: "app/service/ml/models/extratree_model.pkl",
        3: "app/service/ml/models/bagging_model.pkl",
        4: "app/service/ml/models/stacking_model.pkl"
    }

    if model_type not in model_paths:
        raise ValueError(f"❌ Invalid model_type: {model_type}. Must be between 0–4.")

    model_path = model_paths[model_type]
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    return model


import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def inference(model_type):
    """
    Run model inference and full evaluation on test data.
    Supports multiple model types.
    """
    model = load_model(model_type)

    # 3️⃣ Read data
    train_path = "app/storage/uploaded_csvs/cumulative_2025.10.03_05.59.39_train.csv"
    test_path = "app/storage/uploaded_csvs/cumulative_2025.10.03_05.59.39_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 4️⃣ Extract true labels
    if "koi_disposition" not in test_df.columns:
        raise ValueError("❌ 'koi_disposition' column not found in test CSV!")

    y_true = test_df["koi_disposition"]

    # 5️⃣ Drop label column from features
    X_train = train_df.drop(columns=["koi_disposition"], errors="ignore")
    X_test = test_df.drop(columns=["koi_disposition"], errors="ignore")

    # 6️⃣ Align features just in case
    X_test = X_test[X_train.columns]

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
