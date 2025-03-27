# 🚀 SkyBound: Rocket Launch Go/No-Go Prediction System  

## 📌 Project Overview  

SkyBound is a machine learning-powered prediction system that determines whether a rocket launch should proceed or be delayed based on real-time weather conditions. Using historical launch data and weather factors, the model predicts a **Go/No-Go** decision to assist launch providers in making informed scheduling decisions.

---

## 🌟 Features  

- **Historical Data Analysis** – Uses past launch events and weather conditions to train an ML model.  
- **Machine Learning Model** – Implements **XGBoost** for accurate predictions.  
- **Real-Time Weather Integration** – Fetches live weather data from the **Open-Meteo API**.  
- **User Input Support** – Users can select a date (up to 7 days out) for a launch forecast.  
- **Interactive Jupyter Notebook UI** – Provides an easy-to-use interface for predictions.  
- **Visualization & Model Performance**:  
  1. **Confusion Matrix** – Evaluates model accuracy.  
  2. **Feature Importance Graph** – Highlights weather factors affecting launches.  
  3. **Threshold Comparison** – Shows prediction variations at different confidence levels.  

---

## 🛰️ How It Works  

1. **Preprocess Data** – Cleans and normalizes historical launch and weather data.  
2. **Train ML Model** – Uses **XGBoost classification** to predict Go/No-Go.  
3. **Fetch Live Weather** – Pulls real-time weather forecasts for the next 7 days.  
4. **Predict Launch Decision** – Uses trained ML model to classify launch status.  
5. **Log Predictions** – Maintains a log of past forecasts for monitoring and evaluation.  

---

## 🛠️ Tech Stack  

- **Python** (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost)  
- **Jupyter Notebook** – Interactive interface for user input & predictions  
- **Open-Meteo API** – Real-time weather data  
- **PyCharm** – Primary development environment  
- **Git & GitHub** – Version control & collaboration  

---

## 📌 Installation & Setup  

### 1️⃣ Clone the Repository  

```sh
git clone https://github.com/your-username/SkyBound.git
cd SkyBound
```

### 2️⃣ Install Dependencies  

```sh
pip install -r requirements.txt
```

### 3️⃣ Run Data Preprocessing  

```sh
python src/preprocess.py
```

### 4️⃣ Train the Machine Learning Model  

```sh
python src/train_model.py
```

### 5️⃣ Run Predictions via Jupyter Notebook  

```sh
jupyter notebook
```
- Open `SkyBound_Predictions.ipynb`
- Select a **date up to 7 days out** and get a **Go/No-Go** prediction!  

---

## 📊 Visualizations  

SkyBound provides key insights into launch decision factors using three visualizations:  

1. **Confusion Matrix** – Evaluates the accuracy of the ML model.  
2. **Feature Importance** – Identifies the most influential weather conditions.  
3. **Threshold Comparison** – Examines prediction outcomes at different confidence thresholds.  

---

## 🔐 Security Considerations  

- API keys are **not required** for Open-Meteo (no hardcoded credentials).  
- User input is handled securely within the Jupyter Notebook UI.  
- **Error logging & monitoring** are implemented for API failures & prediction tracking.  

---

## 📜 License  

MIT License – Feel free to modify and contribute.  

---

## 🚀 Author  

**Graham Cockerham**  
**Bachelor of Science in Computer Science | Western Governors University**  

---

## 🔹 Contributing  

Pull requests are welcome! For major changes, please open an issue first to discuss your proposal.  

---

This version reflects the **latest changes**, including:
✔ **Switch from Random Forest to XGBoost**  
✔ **Updated API source (Open-Meteo instead of OpenWeatherMap)**  
✔ **Jupyter Notebook UI** instead of CLI/Flask  
✔ **Updated security features & logging**  

