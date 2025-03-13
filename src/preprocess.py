import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess():
    """
    Loads the sample dataset, cleans missing values, normalizes numerical features,
    and saves the processed dataset.
    """
    print("📂 Loading dataset...")

    # Load the dataset
    df = pd.read_csv("../data/sample_launch_data.csv")

    # Check for missing values and fill them if necessary
    if df.isnull().sum().sum() > 0:
        print("⚠️ Missing values detected! Filling missing values...")
        df.fillna(method="ffill", inplace=True)

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ["temperature", "wind_speed", "humidity", "cloud_cover", "pressure"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Ensure target column is present
    if "launch_status" not in df.columns:
        raise ValueError("Missing target column 'launch_status' in dataset!")

    # Save cleaned dataset
    processed_file_path = "../data/processed_launch_data.csv"
    df.to_csv(processed_file_path, index=False)

    print(f"✅ Data preprocessed and saved as '{processed_file_path}'")
    print(df.head())  # Display the first few rows of processed data

if __name__ == "__main__":
    load_and_preprocess()
