import pandas as pd
import os

INPUT_PATH = os.path.join("data", "processed", "processed_data.csv")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "engineered_data.csv")

def load_data(path):
    return pd.read_csv(path)

def create_features(df):
    df["total_screen_time"] = df["social_media_hours"] + df["netflix_hours"]
    df["study_sleep_ratio"] = df["study_hours_per_day"] / df["sleep_hours"]
    df["attendance_study_interaction"] = (
        df["attendance_percentage"] * df["study_hours_per_day"]
    )
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(INPUT_PATH)
    df = create_features(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Feature engineering complete. Saved to", OUTPUT_PATH)

if __name__ == "__main__":
    main()
