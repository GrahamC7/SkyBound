import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
import os

# Ensure the script runs from the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
data_path = os.path.join(script_dir, "..", "data", "processed_data.csv")  # Navigate to data directory

# Load preprocessed data
print("📂 Loading preprocessed data...")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"🚨 ERROR: File not found -> {data_path}")

df = pd.read_csv(data_path)

# Define features and target variable
X = df.drop(columns=["launch_status"])  # Assuming "launch_status" is the target
y = df["launch_status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model (remove `use_label_encoder`)
model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

# Parameter grid (reduced for efficiency)
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 9],
    "n_estimators": [100, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Run GridSearchCV (Reduce number of folds)
print("🔍 Running optimized GridSearchCV for XGBoost...")
grid_search = GridSearchCV(model, param_grid, cv=2, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"✅ Best parameters found: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# Save the best model
model_path = os.path.join(script_dir, "..", "models", "xgboost_best_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
dump(best_model, model_path)
print(f"✅ Model saved successfully at: {model_path}")
