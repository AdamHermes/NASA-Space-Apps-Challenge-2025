from app.routers.merge_csvs import merge_selected_csvs

df = merge_selected_csvs(["test_new_data.csv", "train_new_data.csv"])
print(len(df)) 