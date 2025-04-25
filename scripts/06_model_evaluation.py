import pandas as pd
import os
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "processed", "engineered_data.csv")
MODELS_DIR = os.path.join("outputs", "models")
FIG_DIR = os.path.join("outputs", "figures")
REPORTS_DIR = os.path.join("outputs", "reports")

def load_data(path):
    return pd.read_csv(path)

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X = df.drop(columns=["student_id", "exam_score"])
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    eval_metrics = {}