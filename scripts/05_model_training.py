import pandas as pd
import os
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "processed", "engineered_data.csv")
MODELS_DIR = os.path.join("outputs", "models")
REPORTS_DIR = os.path.join("outputs", "reports")

def load_data(path):
    return pd.read_csv(path)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X = df.drop(columns=["student_id", "exam_score"])
    y = df["exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Linear Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    results["LinearRegression"] = {
        "MSE": mean_squared_error(y_test, y_pred_lr),
        "R2": r2_score(y_test, y_pred_lr)
    }
    joblib.dump(
        lr_pipe,
        os.path.join(MODELS_DIR, "linear_regression_pipeline.pkl")
    )

    # Random Forest + GridSearch
    rf = RandomForestRegressor(random_state=42)
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", rf)
    ])
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20]
    }
    grid = GridSearchCV(rf_pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    results["RandomForestRegressor"] = {
        "BestParams": grid.best_params_,
        "MSE": mean_squared_error(y_test, y_pred_rf),
        "R2": r2_score(y_test, y_pred_rf)
    }
    joblib.dump(
        best_rf,
        os.path.join(MODELS_DIR, "random_forest_pipeline.pkl")
    )

    # save metrics
    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Training complete. Models →", MODELS_DIR, "Metrics →", REPORTS_DIR)

if __name__ == "__main__":
    main()
