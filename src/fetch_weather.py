import requests
import pandas as pd

# Define Cape Canaveral coordinates
latitude = 28.3922
longitude = -80.6077

# Define date range (modify if needed)
start_date = "1957-01-01"
end_date = "2023-12-31"

# Define weather variables
weather_variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "cloudcover",
    "pressure_msl"
]

# Construct API URL
url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}" \
      f"&start_date={start_date}&end_date={end_date}" \
      f"&hourly={','.join(weather_variables)}" \
      f"&timezone=America/New_York"

# Fetch weather data
print("🌎 Fetching historical weather data...")
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print("✅ Successfully connected to Open-Meteo API")

    # Check full response before processing
    data = response.json()
    print("🔍 JSON Response:", data)  # Print the full API response

    # Ensure 'hourly' exists before processing
    if "hourly" in data:
        print("✅ 'hourly' data found!")

        # Convert to DataFrame
        weather_data = pd.DataFrame(data["hourly"])

        # Ensure 'time' column exists before processing
        if "time" in weather_data:
            weather_data["date"] = pd.to_datetime(weather_data["time"]).dt.date  # Extract only the date

            # Save the data
            file_path = "../data/historical_weather.csv"
            weather_data.to_csv(file_path, index=False)
            print(f"✅ Weather data saved as '{file_path}'")
        else:
            print("❌ 'time' column missing in API response. Check the JSON structure.")
    else:
        print("❌ 'hourly' data missing in API response. API may have changed.")
else:
    print(f"❌ Failed to fetch data. HTTP Status Code: {response.status_code}")
    print("Response Text:", response.text)
