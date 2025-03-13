import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 500

# Generate random weather conditions
np.random.seed(42)
temperature = np.random.uniform(10, 35, num_samples)  # Temperature in °C
wind_speed = np.random.uniform(0, 20, num_samples)    # Wind speed in m/s
humidity = np.random.uniform(30, 90, num_samples)     # Humidity percentage
cloud_cover = np.random.uniform(0, 100, num_samples)  # Cloud cover percentage
pressure = np.random.uniform(950, 1050, num_samples)  # Atmospheric pressure in hPa

# Simulate launch decisions based on weather conditions
launch_status = [
    1 if (temp > 15 and wind < 15 and humidity < 80 and cloud < 75 and pressure > 980) else 0
    for temp, wind, humidity, cloud, pressure in zip(temperature, wind_speed, humidity, cloud_cover, pressure)
]

# Create DataFrame
df = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "humidity": humidity,
    "cloud_cover": cloud_cover,
    "pressure": pressure,
    "launch_status": launch_status  # 1 = Go, 0 = No-Go
})

# Save to CSV
df.to_csv("../data/sample_launch_data.csv", index=False)
print("✅ Sample dataset generated and saved as 'sample_launch_data.csv'")
