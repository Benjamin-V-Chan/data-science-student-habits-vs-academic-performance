# - import pandas, os, joblib, json
#   from sklearn.model_selection import train_test_split, GridSearchCV
#   from sklearn.linear_model import LinearRegression
#   from sklearn.ensemble import RandomForestRegressor
#   from sklearn.pipeline import Pipeline
#   from sklearn.preprocessing import StandardScaler
#   from sklearn.metrics import mean_squared_error, r2_score
# - define load_data(path)
# - in main():
#     • ensure outputs/models & outputs/reports exist
#     • load data/processed/engineered_data.csv
#     • split X/y, then train_test_split(random_state=42)
#     • build & fit:
#         – LinearRegression pipeline
#         – RandomForestRegressor pipeline via GridSearchCV
#     • evaluate on test set
#     • save both pipelines to outputs/models/
#     • save a JSON of metrics to outputs/reports/model_metrics.json
