import os






def get_current_csv_files():
    return os.listdir("app/storage/uploaded_csvs")



def get_current_models():
    return os.listdir("app/storage/weights")



