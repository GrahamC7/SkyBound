import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

data_path = "../data/final_data.csv"
absolute_path = os.path.abspath(data_path)

print(f"Looking for file at: {absolute_path}")

# check if the file exists before loading
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {absolute_path}")

df = pd.read_csv(data_path)
print("File loaded successfully!")

# load merged dataset
data_path = "../data/final_data.csv"
print("Loading data...")
df = pd.read_csv(data_path)

# step 1: check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# step 2: handle missing values
df.ffill(inplace=True)

# define target column
target_column = "launch_status"

# step 3: convert categorical target variable to binary
if target_column in df.columns:
    df[target_column] = np.where(df[target_column] > 0, 1, 0)  # ensure binary format

# debugging check
print(f"Fixed '{target_column}' values: {df[target_column].unique()}")

# step 4: normalize numerical features (excluding target column)
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
numeric_features = numeric_features.drop(target_column, errors="ignore")  # exclude target column

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# step 5: save preprocessed data
preprocessed_data_path = "../data/preprocessed_data.csv"
df.to_csv(preprocessed_data_path, index=False)

print(f"Data preprocessing complete! Saved as '{preprocessed_data_path}'")
