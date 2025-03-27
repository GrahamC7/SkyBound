import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# load trained xgboost model
model_path = "../models/xgboost_best_model.pkl"
print("Loading trained XGBoost model...")
xgb_model = joblib.load(model_path)

# load test dataset
print("Loading test dataset...")
data_path = "../data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# drop target column
X_test = df.drop(columns=["launch_status"])
y_test = df["launch_status"]

# ensure categorical columns are properly one-hot encoded
categorical_cols = ["launch_location", "Status Mission", "time"]
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# get the feature names used in training
expected_features = xgb_model.get_booster().feature_names

# reorder and fill missing columns with 0
X_test = X_test.reindex(columns=expected_features, fill_value=0)

# generate predictions
y_pred = xgb_model.predict(X_test)

# create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No-Go", "Go"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix for Rocket Launch Prediction")
plt.savefig("../visualizations/confusion_matrix.png", dpi=300)
print("Confusion matrix saved as '../visualizations/confusion_matrix.png'")
plt.show()
