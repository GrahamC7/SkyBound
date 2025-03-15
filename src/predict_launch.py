import requests
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_path = "../models/rocket_launch_model.pkl"
print("📂 Loading trained model...")
model = joblib.load(model_path)

# Open-Meteo API Endpoint for real-time weather data
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
latitude, longitude = 28.3922, -80.6077  # Cape Canaveral coordinates

# Define the weather variables needed for prediction
weather_variables = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "windspeed_10m", "windgusts_10m", "winddirection_10m",
    "cloudcover", "pressure_msl"
]

# Fetch real-time weather data
def fetch_weather_data():
    print("🌎 Fetching real-time weather data...")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true"
    }
    response = requests.get(OPEN_METEO_URL, params=params)
    data = response.json()

    if "current_weather" not in data:
        raise ValueError("❌ Failed to fetch weather data!")

    # Extract the required features (use .get() to avoid KeyErrors)
    weather = data["current_weather"]
    print("\n🌎 Raw API Response:", data)  # Debugging: Print full API response

    formatted_data = {
        "temperature_2m": weather.get("temperature", np.nan),
        "relative_humidity_2m": 60.0,  # Default to 60% if missing (approximate value)
        "precipitation": weather.get("precipitation", 0.0),  # Default to 0.0 if missing
        "windspeed_10m": weather.get("windspeed", np.nan),
        "windgusts_10m": 15.0,  # Estimated default (replace with better source)
        "winddirection_10m": weather.get("winddirection", np.nan),
        "cloudcover": 50.0,  # Default to 50% if missing
        "pressure_msl": 1013.25  # Default standard atmospheric pressure in hPa
    }

    # Convert to DataFrame
    weather_df = pd.DataFrame([formatted_data])
    return weather_df


# Run Prediction
def predict_launch():
    weather_df = fetch_weather_data()

    print("\n🌤️ Current Weather Data for Prediction:")
    print(weather_df)

    # Fill missing values with the mean (since some weather variables are missing)
    weather_df.fillna(weather_df.mean(), inplace=True)

    # Make prediction
    prediction = model.predict(weather_df)[0]
    confidence = model.predict_proba(weather_df)[0]

    # Convert prediction to label
    decision = "GO" if prediction == 1 else "NO-GO"
    confidence_score = max(confidence) * 100

    print(f"\n🚀 **Launch Decision:** {decision}")
    print(f"📊 **Confidence Score:** {confidence_score:.2f}%")

# Run the prediction script
if __name__ == "__main__":
    predict_launch()
