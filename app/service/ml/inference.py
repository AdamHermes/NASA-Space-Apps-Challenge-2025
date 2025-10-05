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
from ..data.data_manage import merge_selected_csvs_to_inference, merge_selected_csvs

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

def process_inference_data(final_data):
    df = final_data

    # ======================
    # Step 1: Remove unnecessary columns
    # ======================
    cols = ['koi_disposition', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
        'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2',
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact',
        'koi_impact_err1', 'koi_impact_err2', 'koi_duration',
        'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1',
        'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
        'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
        'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_steff',
        'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1',
        'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra',
        'dec', 'koi_kepmag']
    
    # ======================
    # Step 2: Keep only candidate and confirmed
    # ======================
    df = df[df["koi_disposition"].isin(["CANDIDATE", "CONFIRMED"])]
    print("After filtering dispositions:", df["koi_disposition"].value_counts())

    # ======================
    # Step 3: Encode target column
    # ======================
    df["koi_disposition"] = df["koi_disposition"].map({"CANDIDATE": 1, "CONFIRMED": 0})

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in CSV: {missing_cols}")

    df = df[cols]
    print("After column removal:", df.shape)

    # ======================
    # Step 4: Handle koi_tce_delivname
    # ======================
    if "koi_tce_delivname" in df.columns:
        # Fill missing values with the mode (most frequent value)
        df["koi_tce_delivname"].fillna(df["koi_tce_delivname"].mode()[0], inplace=True)

        # Create Boolean dummy columns for only the two categories you care about
        df["koi_tce_delivname_q1_q17_dr24_tce"] = df["koi_tce_delivname"] == "q1_q17_dr24_tce"
        df["koi_tce_delivname_q1_q17_dr25_tce"] = df["koi_tce_delivname"] == "q1_q17_dr25_tce"

        # Optionally drop the original column
        df.drop(columns=["koi_tce_delivname"], inplace=True)
    df = df.dropna().reset_index(drop=True)

    print(df.columns)
    return df

def inference_list_csvs(model_type, model_name, list_csv_names):
    """
    Run model inference and full evaluation on test data.
    Supports multiple model types.
    """
    model_path = os.path.join("app/storage/weights", model_type , model_name)
    model = joblib.load(model_path)
    

    # final_data = merge_selected_csvs_to_inference(list_csv_names)
    loaded_final_data = merge_selected_csvs(list_csv_names)
    final_data = process_inference_data(final_data=loaded_final_data)
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
    scaler_path = os.path.join("app/storage/scalers", str(model_type), f"{just_model_name}_scaler.pkl")

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
    print(X_test)

    # 7️⃣ Predict
    y_pred = model.predict(X_test)   # ✅ FIXED: removed `.values`
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
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    # cm = confusion_matrix(y_true, y_pred)
    # cm_df = pd.DataFrame(cm, index=sorted(set(y_true)), columns=sorted(set(y_true)))

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
    pass

