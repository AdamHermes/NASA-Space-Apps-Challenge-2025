from app.service.data.process_koi import process_koi

df = process_koi(["test_new_data.csv", "train_new_data.csv"])
print(len(df)) 