import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
data_path = "../data/preprocessed_data.csv"
print("📂 Loading preprocessed data...")
df = pd.read_csv(data_path)

# Identify the target column
target_column = "launch_status"  # Ensure this is the correct column representing Go/No-Go

# Convert categorical target variable to numeric
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Drop non-numeric columns
non_numeric_columns = ["date", "launch_location", "Status Mission", "time"]
df = df.drop(columns=non_numeric_columns, errors="ignore")

# Separate features (X) and target variable (y)
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    # Train a Random Forest Classifier
    print("🌲 Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.2f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🔄 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Define model directory and ensure it exists
    model_dir = "../models"
    model_path = f"{model_dir}/rocket_launch_model.pkl"
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved as '{model_path}'")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
