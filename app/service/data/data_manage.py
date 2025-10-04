import os
import json






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


