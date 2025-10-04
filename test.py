import requests

url = "http://127.0.0.1:8000/train/retrain/"

data = {
    "filename": "new_data.csv",
    "models": ["RandomForest", "AdaBoost"],
    "learning_rate": 0.5,
    "n_estimators": 100,
    "max_depth": 10
}

response = requests.post(url, data=data)
print(response.json())




