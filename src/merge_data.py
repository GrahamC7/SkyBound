import pandas as pd

# Load rocket launch data
launch_data_path = "../data/processed_launch_data.csv"
weather_data_path = "../data/historical_weather.csv"

print("🚀 Loading rocket launch data...")
launch_df = pd.read_csv(launch_data_path)

print("🌎 Loading historical weather data...")
weather_df = pd.read_csv(weather_data_path)

# Convert 'date' columns to datetime format
launch_df["date"] = pd.to_datetime(launch_df["date"]).dt.date  # Remove timezone
weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date  # Remove timezone

# Merge datasets on the 'date' column (matching weather to launches)
print("🔄 Merging weather data with rocket launch data...")
merged_df = pd.merge(launch_df, weather_df, on="date", how="left")

# Save the final dataset
final_data_path = "../data/final_data.csv"
merged_df.to_csv(final_data_path, index=False)

print(f"✅ Final dataset saved as '{final_data_path}'")
