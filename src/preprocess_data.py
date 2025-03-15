import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load merged dataset
data_path = "../data/final_data.csv"
print("📂 Loading data...")
df = pd.read_csv(data_path)

# Step 1: Check for missing values
print("\n🔍 Checking for missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Step 2: Handle missing values (impute or remove)
df.fillna(method="ffill", inplace=True)  # Forward fill missing values

# Define the target column
target_column = "launch_status"

# Step 3: Convert categorical target variable to binary (if applicable)
if target_column in df.columns:
    df[target_column] = np.where(df[target_column] > 0, 1, 0)  # Ensure binary format

# Debugging check
print(f"✅ Fixed '{target_column}' values: {df[target_column].unique()}")

# Step 4: Normalize numerical features (EXCLUDING target column)
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
numeric_features = numeric_features.drop(target_column, errors="ignore")  # Exclude target column

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Step 5: Save preprocessed data
preprocessed_data_path = "../data/preprocessed_data.csv"
df.to_csv(preprocessed_data_path, index=False)

print(f"✅ Data preprocessing complete! Saved as '{preprocessed_data_path}'")
