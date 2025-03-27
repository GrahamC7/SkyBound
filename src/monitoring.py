import logging
import os
import datetime
import pandas as pd

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/system_monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to log errors
def log_error(message):
    """Logs an error message with a timestamp."""
    logging.error(message)
    print(f"ERROR: {message}")  # also print for visibility


# Function to log successful predictions
def log_prediction(date, decision, confidence):
    """Logs predictions for future analysis."""
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_predicted": date,
        "launch_decision": decision,
        "confidence": confidence
    }

    # Append to CSV file
    log_file = "logs/prediction_log.csv"
    df = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, mode='w', header=True, index=False)

    print(f"Prediction Logged: {log_entry}")


# Test Logging (Run this once to verify)
if __name__ == "__main__":
    log_error("Test error logging - API request failed!")
    log_prediction("2025-03-25", "GO", 96.7)
