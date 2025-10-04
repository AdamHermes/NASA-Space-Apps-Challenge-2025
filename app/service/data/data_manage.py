import os
import json
from pathlib import Path
import pandas as pd





def get_current_csv_files():
    return os.listdir("app/storage/uploaded_csvs")



def get_current_models():
    base_dir = "app/storage/weights"
    models_info = {}

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if os.path.isdir(model_path):
            weights = [
                f for f in os.listdir(model_path)
                if os.path.isfile(os.path.join(model_path, f))
            ]
            models_info[model_name] = weights

    return json.dumps(models_info, indent=4)





def merge_selected_csvs(csv_files: list[str]) -> pd.DataFrame:
    """
    Merge selected CSV files from UPLOAD_DIR and return as a pandas DataFrame.
    
    Args:
        csv_files (list[str]): List of CSV filenames to merge.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    merged_df = pd.DataFrame()
    UPLOAD_DIR = Path("app/storage/uploaded_csvs")

    for filename in csv_files:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found in uploaded CSVs")
        
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    return merged_df



