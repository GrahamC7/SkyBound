import requests
import joblib
import pandas as pd
import numpy as np
import os

# get absolute path to the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "xgboost_best_model.pkl")

print("Loading trained model...")
model = joblib.load(model_path)

# open-meteo api for real-time weather data
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
latitude, longitude = 28.3922, -80.6077  # cape canaveral coordinates

# define weather variables needed for prediction
weather_variables = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "windspeed_10m", "windgusts_10m", "winddirection_10m",
    "cloudcover", "pressure_msl"
]

# fetch real-time weather data
def fetch_weather_data():
    print("Fetching real-time weather data...")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true"
    }
    response = requests.get(OPEN_METEO_URL, params=params)
    data = response.json()

    if "current_weather" not in data:
        raise ValueError("Failed to fetch weather data!")

    # extract required features
    weather = data["current_weather"]
    print("\nRaw API Response:", data)  # debugging: print full api response

    formatted_data = {
        "temperature_2m": weather.get("temperature", np.nan),
        "relative_humidity_2m": 60.0,  # default to 60% if missing
        "precipitation": weather.get("precipitation", 0.0),  # default to 0.0 if missing
        "windspeed_10m": weather.get("windspeed", np.nan),
        "windgusts_10m": 15.0,  # estimated default
        "winddirection_10m": weather.get("winddirection", np.nan),
        "cloudcover": 50.0,  # default to 50% if missing
        "pressure_msl": 1013.25  # default standard atmospheric pressure in hpa
    }

    # convert to dataframe
    weather_df = pd.DataFrame([formatted_data])
    return weather_df

# run prediction
def predict_launch():
    weather_df = fetch_weather_data()

    print("\nCurrent Weather Data for Prediction:")
    print(weather_df)

    # fill missing values with the mean
    weather_df.fillna(weather_df.mean(), inplace=True)

    # make prediction
    prediction = model.predict(weather_df)[0]
    confidence = model.predict_proba(weather_df)[0]

    # convert prediction to label
    decision = "GO" if prediction == 1 else "NO-GO"
    confidence_score = max(confidence) * 100

    print(f"\nLaunch Decision: {decision}")
    print(f"Confidence Score: {confidence_score:.2f}%")

# run the prediction script
if __name__ == "__main__":
    predict_launch()

