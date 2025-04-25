import pandas as pd
import os

RAW_DATA_PATH = os.path.join("data", "raw", "student_habits_performance.csv")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "raw_data.csv")

def load_data(path):
    return pd.read_csv(path)

def save_data(df, path):
    df.to_csv(path, index=False)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = load_data(RAW_DATA_PATH)
    print("Loaded raw data with shape:", df.shape)
    save_data(df, OUTPUT_PATH)
    print("Saved raw data to", OUTPUT_PATH)

if __name__ == "__main__":
    main()
