# - import pandas, numpy, os
# - define load_data(path)
# - define handle_missing(df):
#     • fill numeric NaNs with median
#     • fill categorical NaNs with mode
# - define encode_categorical(df):
#     • map ordinal features (diet_quality, internet_quality, parental_education_level)
#     • one-hot encode gender, part_time_job, extracurricular_participation
# - define save_data(df, path)
# - in main():
#     • load data/processed/raw_data.csv
#     • handle_missing → encode_categorical
#     • save to data/processed/processed_data.csv
