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
    for fname in os.listdir(MODELS_DIR):
        model_name = fname.replace("_pipeline.pkl", "")
        model = joblib.load(os.path.join(MODELS_DIR, fname))
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        eval_metrics[model_name] = {"MSE": mse, "R2": r2}

        # Pred vs Actual
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} Actual vs Predicted")
        plt.savefig(os.path.join(FIG_DIR, f"{model_name}_pred_vs_actual.png"))
        plt.close()

        # Residuals
        residuals = y_test - y_pred
        plt.figure()
        plt.hist(residuals, bins=30)
        plt.xlabel("Residual")
        plt.title(f"{model_name} Residuals")
        plt.savefig(os.path.join(FIG_DIR, f"{model_name}_residuals.png"))
        plt.close()

    with open(os.path.join(REPORTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)

    print("Evaluation complete. Figures →", FIG_DIR, "Reports →", REPORTS_DIR)

if __name__ == "__main__":
    main()
