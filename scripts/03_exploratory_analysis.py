import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = os.path.join("data", "processed", "processed_data.csv")
FIG_DIR = os.path.join("outputs", "figures")

def load_data(path):
    return pd.read_csv(path)

def plot_distribution(df, col, out_dir):
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(os.path.join(out_dir, f"{col}_dist.png"))
    plt.close()

def plot_correlation(df, out_dir):
    plt.figure(figsize=(10,8))
    corr = df.select_dtypes(include=["float64", "int64"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_data(INPUT_PATH)
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        plot_distribution(df, col, FIG_DIR)
    plot_correlation(df, FIG_DIR)
    print("EDA complete. Figures in", FIG_DIR)

if __name__ == "__main__":
    main()
