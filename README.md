🚀 SkyBound: Rocket Launch Go/No-Go Prediction System

📌 Project Overview

SkyBound is a machine learning-powered prediction system that determines whether a rocket launch should proceed or be delayed based on real-time weather conditions. Using historical launch data and weather factors, the model predicts a Go/No-Go decision to assist launch providers in making informed scheduling decisions.

🌟 Features

Historical Data Analysis: Uses past launch events and weather conditions to train an ML model.

Machine Learning Model: Implements Random Forest/XGBoost classification for predictions.

Real-Time Weather Integration: Fetches live weather data from the OpenWeatherMap API.

User Input Support: Users can manually enter weather conditions or use live API data.

Interactive UI (CLI or Flask): Provides a simple command-line interface for predictions (optional Flask web app).

Visualization & Model Performance:

Confusion Matrix for model accuracy

Feature Importance Graph

Weather Trends vs. Launch Outcomes

🛰️ How It Works

Preprocess Data: Cleans and normalizes historical launch and weather data.

Train ML Model: Uses a classification algorithm to predict Go/No-Go.

Fetch Live Weather: Pulls real-time weather data from OpenWeatherMap.

Predict Launch Decision: Uses trained ML model to classify launch status.

🛠️ Tech Stack

Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Flask)

Git & GitHub for version control

OpenWeatherMap API for real-time weather data

PyCharm for development

📌 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/SkyBound.git
cd SkyBound

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Data Preprocessing

python src/preprocess.py

4️⃣ Train the Machine Learning Model

python src/train_model.py

5️⃣ Run Predictions

python src/predict.py

📜 License

MIT License - Feel free to modify and contribute.

🚀 Author

Graham CockerhamBachelor of Science in Computer Science | Western Governors University

🔹 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.