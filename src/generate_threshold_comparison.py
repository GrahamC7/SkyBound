import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load trained XGBoost model
model_path = "../models/xgboost_best_model.pkl"
print("Loading trained XGBoost model...")
xgb_model = joblib.load(model_path)

# Load test dataset
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Ensure only numeric columns are used
numeric_features = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "windspeed_10m", "windgusts_10m", "winddirection_10m",
    "cloudcover", "pressure_msl"
]
X_test = df[numeric_features]  # Select only numeric features
y_test = df["launch_status"]

# Get model probabilities
y_probs = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for "Go"

# Define thresholds
thresholds = [0.45, 0.50, 0.55]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, threshold in enumerate(thresholds):
    y_pred = (y_probs >= threshold).astype(int)  # Apply threshold
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No-Go", "Go"])
    disp.plot(ax=axes[i], cmap="Blues", values_format="d")
    axes[i].set_title(f"XGBoost (Threshold = {threshold})")

plt.tight_layout()
plt.savefig("../visualizations/threshold_comparison.png", dpi=300)
print("✅ Threshold comparison saved as '../visualizations/threshold_comparison.png'")
plt.show()
