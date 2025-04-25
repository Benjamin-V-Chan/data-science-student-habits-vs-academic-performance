# - import pandas, os, joblib, json, matplotlib.pyplot
#   from sklearn.model_selection import train_test_split
#   from sklearn.metrics import mean_squared_error, r2_score
# - define load_data(path)
# - in main():
#     • ensure outputs/figures & outputs/reports exist
#     • load engineered_data.csv; split X/y & train_test_split(same seed)
#     • for each saved model in outputs/models/:
#         – load model
#         – predict on X_test
#         – compute MSE & R²
#         – scatter plot actual vs predicted → save to outputs/figures/
#         – histogram of residuals → save
#     • save a JSON of evaluation metrics to outputs/reports/evaluation_metrics.json
