# your_model_module.py
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    StackingClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb

import joblib
from typing import Tuple
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, save_path=None):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    metrics = evaluate_model(rf, X_test, y_test)
    if save_path:
        joblib.dump(rf, save_path)
    return rf, metrics

def train_adaboost(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=1.0, save_path=None):
    ab = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    ab.fit(X_train, y_train)
    metrics = evaluate_model(ab, X_test, y_test)
    if save_path:
        joblib.dump(ab, save_path)
    return ab, metrics

def train_stacking(X_train, y_train, X_test, y_test, save_path=None):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('ab', AdaBoostClassifier(n_estimators=50, random_state=42))
    ]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    stack.fit(X_train, y_train)
    metrics = evaluate_model(stack, X_test, y_test)
    if save_path:
        joblib.dump(stack, save_path)
    return stack, metrics

def train_random_subspace(X_train, y_train, X_test, y_test, n_estimators=100, max_features=0.5, save_path=None):
    rsm = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=42
    )
    rsm.fit(X_train, y_train)
    metrics = evaluate_model(rsm, X_test, y_test)
    if save_path:
        joblib.dump(rsm, save_path)
    return rsm, metrics

def train_extra_trees(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, save_path=None):
    et = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    et.fit(X_train, y_train)
    metrics = evaluate_model(et, X_test, y_test)
    if save_path:
        joblib.dump(et, save_path)
    return et, metrics

def train_lightgbm(X_train, y_train, X_test, y_test,  n_estimators=100, learning_rate=0.1, save_path=None):
    lgbm = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    lgbm.fit(X_train, y_train)
    metrics = evaluate_model(lgbm, X_test, y_test)
    if save_path:
        joblib.dump(lgbm, save_path)
    return lgbm, metrics
# your_model_module.py



def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    learning_rate: float = None,
    n_estimators: int = None,
    max_depth: int = None
) -> Tuple[object, dict]:
    """
    Dispatch function to call the appropriate model training function based on model_name.
    
    Returns:
        model: trained model objectc
        metrics: dict containing accuracy, confusion matrix, classification report
    """
    model_name = model_name.lower()  # normalize string

    if model_name == "randomforest":
        model, metrics = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators or 100,
            max_depth=max_depth,
        )
    elif model_name == "adaboost":
        model, metrics = train_adaboost(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators or 100,
            learning_rate=learning_rate or 1.0,
        )
    elif model_name == "stacking":
        model, metrics = train_stacking(X_train, y_train, X_test, y_test)
    elif model_name in ["bagging", "randomsubspace", "rsm"]:
        model, metrics = train_random_subspace(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators or 100,
            max_features=0.5
        )
    elif model_name in ["extratrees", "extremelyrandomizedtrees", "et"]:
        model, metrics = train_extra_trees(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators or 100,
            max_depth=max_depth,
        )
    elif model_name == "lightgbm":
        model,metrics = train_lightgbm(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators or 100,
            learning_rate=learning_rate or 0.1,
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'")

    return model, metrics
