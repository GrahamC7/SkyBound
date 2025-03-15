import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load preprocessed data
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Load trained model
model_path = "../models/rocket_launch_model.pkl"

# ✅ Ensure the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

print("📂 Loading trained model...")
model = joblib.load(model_path)

# Define the target column
target_column = "launch_status"

# Drop non-numeric columns
non_numeric_columns = ["date", "launch_location", "Status Mission", "time"]
df = df.drop(columns=non_numeric_columns, errors="ignore")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# ✅ Ensure y is binary (0 or 1)
y = np.where(y > 0, 1, 0)

# Predict the test set
y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ✅ Debugging: Check prediction values
print(f"✅ Unique values in y_pred (predicted labels): {np.unique(y_pred)}")

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Ensure labels match available classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No-Go", "Go"])

# ✅ Fix: Close existing figures to avoid blank Figure 1
plt.close('all')

# Save confusion matrix
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix for Rocket Launch Prediction")

# Define output path
output_path = "../visualizations/confusion_matrix.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()  # ✅ Close the figure after saving
print(f"✅ Confusion matrix saved as '{output_path}'")
