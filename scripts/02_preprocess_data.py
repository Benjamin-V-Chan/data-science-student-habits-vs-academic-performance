import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "processed", "raw_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "processed_data.csv")

def load_data(path):
    return pd.read_csv(path)

def handle_missing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    df['diet_quality'] = df['diet_quality'].map({'Poor':0, 'Fair':1, 'Good':2})
    df['internet_quality'] = df['internet_quality'].map({'Poor':0, 'Average':1, 'Good':2})
    df['parental_education_level'] = df['parental_education_level'].map({
        'None':0, 'High School':1, 'Bachelor':2, 'Master':3
    })
    df = pd.get_dummies(
        df,
        columns=['gender', 'part_time_job', 'extracurricular_participation'],
        drop_first=True
    )
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

def main():
    df = load_data(RAW_PATH)
    df = handle_missing(df)
    df = encode_categorical(df)
    save_data(df, OUTPUT_PATH)
    print("Preprocessing complete. Saved to", OUTPUT_PATH)

if __name__ == "__main__":
    main()
