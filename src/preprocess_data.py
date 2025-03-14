import pandas as pd
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

# Step 3: Convert categorical data to numerical
if 'go_nogo' in df.columns:  # Assuming 'go_nogo' is the target variable
    label_encoder = LabelEncoder()
    df['go_nogo'] = label_encoder.fit_transform(df['go_nogo'])  # Encode Yes/No to 1/0

# Step 4: Normalize numerical features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Step 5: Save preprocessed data
preprocessed_data_path = "../data/preprocessed_data.csv"
df.to_csv(preprocessed_data_path, index=False)

print(f"✅ Data preprocessing complete! Saved as '{preprocessed_data_path}'")
