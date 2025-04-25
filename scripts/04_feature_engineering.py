# - import pandas, os
# - define load_data(path)
# - define create_features(df):
#     • total_screen_time = social_media_hours + netflix_hours
#     • study_sleep_ratio = study_hours_per_day / sleep_hours
#     • attendance_study_interaction = attendance_percentage * study_hours_per_day
# - in main():
#     • load data/processed/processed_data.csv
#     • df = create_features(df)
#     • save to data/processed/engineered_data.csv
