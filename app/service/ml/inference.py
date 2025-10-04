import pandas as pd
from io import BytesIO
from .split import split
import joblib


def inference():
    ab_loaded = joblib.load("./app/storage/adaboost_model.pkl")
    print(ab_loaded)

