from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .data_manage import merge_selected_csvs

CSV_UPLOAD_DIR = Path("app/storage/uploaded_csvs")
PROCESS_CSV_DIR = Path("app/storage/processed_csvs")

def get_dataset_stats(df, label_col="koi_disposition"):
    stats = {
        "num_samples": len(df),
        "num_features": df.shape[1] - 1,  # excluding label
        "class_counts": df[label_col].value_counts().to_dict(),
        "class_percentage": (df[label_col].value_counts(normalize=True) * 100).round(2).to_dict()
    }
    return stats

def process_koi(csv_name):
    try:
        if len(csv_name) > 1:
            print("[INFO] Processing csvs:", csv_name)
            original_df = merge_selected_csvs(csv_files=csv_name)
            csv_path = CSV_UPLOAD_DIR / "merged_data.csv"
        else: 
            csv_name = csv_name[0]
            csv_path = CSV_UPLOAD_DIR / csv_name
            if not csv_path.exists():
                raise FileNotFoundError(f"{csv_name} not found in {CSV_UPLOAD_DIR}")

            print("[INFO] Processing csv_path:", csv_path)
            original_df = pd.read_csv(csv_path, comment="#")
        print("Original shape:", original_df.shape)

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

        missing_cols = [c for c in cols if c not in original_df.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in CSV: {missing_cols}")

        df = original_df[cols]
        print("After column removal:", df.shape)

        # ======================
        # Step 2: Keep only candidate and confirmed
        # ======================
        df = df[df["koi_disposition"].isin(["CANDIDATE", "CONFIRMED"])]
        print("After filtering dispositions:", df["koi_disposition"].value_counts())

        # ======================
        # Step 3: Encode target column
        # ======================
        df["koi_disposition"] = df["koi_disposition"].map({"CANDIDATE": 1, "CONFIRMED": 0})

        # ======================
        # Step 4: Handle koi_tce_delivname
        # ======================
        if "koi_tce_delivname" in df.columns:
            df["koi_tce_delivname"].fillna(df["koi_tce_delivname"].mode()[0], inplace=True)
            df = pd.get_dummies(df, columns=["koi_tce_delivname"], drop_first=True)

        # ======================
        # Step 5: Split into X and y
        # ======================

        X = df.drop(columns=["koi_disposition"])
        y = df["koi_disposition"]
        df_combined = pd.concat([X, y], axis=1)
        df_combined.dropna(inplace=True)
        X = df_combined.drop(columns=["koi_disposition"])
        y = df_combined["koi_disposition"]

        # ======================
        # Step 6: Scale features
        # ======================
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ======================
        # Step 7 & 8: Train-test split
        # ======================
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Train shape:", X_train.shape, y_train.shape)
        print("Test shape:", X_test.shape, y_test.shape)

        # Save train and test as CSV
        train_df = pd.DataFrame(X_train, columns=X.columns)
        train_df["koi_disposition"] = y_train.values

        test_df = pd.DataFrame(X_test, columns=X.columns)
        test_df["koi_disposition"] = y_test.values

        train_stats = get_dataset_stats(train_df)
        test_stats = get_dataset_stats(test_df)

        original_file = PROCESS_CSV_DIR / f"{csv_path.stem}_processed.csv"
        train_file = PROCESS_CSV_DIR / f"{csv_path.stem}_train.csv"
        test_file = PROCESS_CSV_DIR / f"{csv_path.stem}_test.csv"

        df_combined.to_csv(original_file, index=False)
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print("[INFO] Train and test files saved:")
        print("Train:", train_file)
        print("Test:", test_file)

        return {
            "train_filename": f"{csv_path.stem}_train.csv",
            "train_filepath": str(train_file),
            "train_stats": train_stats,
            "test_filename": f"{csv_path.stem}_test.csv",
            "test_filepath": str(test_file),
            "test_stats": test_stats,
            "all_filename": f"{csv_path.stem}_processed.csv",
            "train_head": train_df.head().to_dict(orient="records"),
            "test_head": test_df.head().to_dict(orient="records"),
        }

    except FileNotFoundError as e:
        print("[ERROR]", e)
        return {"error": str(e)}
    except KeyError as e:
        print("[ERROR]", e)
        return {"error": str(e)}
    except Exception as e:
        print("[ERROR] Unexpected error:", e)
        return {"error": str(e)}
