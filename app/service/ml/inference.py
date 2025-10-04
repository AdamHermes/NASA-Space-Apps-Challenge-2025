import pandas as pd
from io import BytesIO
from .split import split
import joblib
import csv


def inference():
    ab_loaded = joblib.load("app/service/ml/models/adaboost_model.pkl")
    
    return ab_loaded
