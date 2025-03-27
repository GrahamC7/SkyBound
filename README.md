# 🚀 SkyBound: Rocket Launch Go/No-Go Prediction System

## 📌 Overview

SkyBound is a machine learning-powered application that predicts whether a rocket launch should proceed or be delayed based on real-time or forecasted weather conditions. The system leverages historical launch and weather data to train a model capable of issuing **Go/No-Go** decisions along with confidence scores.

---

## 🌟 Features

- **Go/No-Go Launch Decision** based on live or forecasted weather
- **XGBoost Machine Learning Model** trained on NASA and NOAA historical data
- **Two Usage Modes**:
  - 📆 **Forecast-based via SkyBound\_Predictions.ipynb**
  - ⌨️ **Real-time/manual input via predict\_launch.ipynb**
- **Three Visualizations**:
  1. Confusion Matrix
  2. Feature Importance
  3. Threshold Comparison
- **CLI Access** via `main.py`
- **Built-in Logging** of predictions and API errors

---

## 🛠️ System Requirements

- Windows 10
- Python 3.x
- Jupyter Notebook
- Python packages listed in `requirements.txt`

---

## 📦 Installation Instructions

1. **Clone or Download the Repository**

   ```bash
   git clone https://github.com/GrahamC7/Skybound.git
   cd Skybound
   ```

   Or download ZIP via GitHub and extract it.

2. **Install Python & pip**\
   Download from [python.org](https://www.python.org/downloads/) and ensure `pip` is included.

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter Notebook (if needed)**

   ```bash
   pip install notebook
   ```

5. **Install XGBoost (if needed)**

   ```bash
   pip install xgboost
   ```

---

## 🚀 How to Run the Application

### Option 1: **Command-Line Interface**

```bash
python main.py
```

- Loads pre-trained model
- Fetches current weather for Cape Canaveral
- Outputs Go/No-Go prediction and confidence score

> 💡 If using a virtual environment:

```bash
.\.venv\Scripts\activate
python main.py
```

---

### Option 2: **Jupyter Notebook Interface**

#### 📆 `SkyBound_Predictions.ipynb`

- Uses Open-Meteo API to pull 7-day forecast
- Interactive date slider for future launch days
- Produces Go/No-Go prediction with confidence score

#### ⌨️ `predict_launch.ipynb`

- Prompts user to choose:
  - 🛰️ Use real-time weather data
  - ✍️ Manually input weather parameters
- Displays launch decision + logs the prediction

> Both notebooks require executing cells in order after opening in Jupyter.

---

## 📊 Visualizations

Included in the project are:

- **Confusion Matrix** – Classification accuracy overview
- **Feature Importance Plot** – Shows weight of each weather factor
- **Threshold Comparison Graph** – Varying cutoff impact on results

These visuals support transparency and help fine-tune launch risk tolerance.

---

## 🐞 Troubleshooting

- **Python not recognized?** Add Python to your PATH or reinstall.
- **ModuleNotFoundError?** Run `pip install xgboost` or missing package.
- **Jupyter not found?** Install with `pip install notebook`.
- **Kernel issues?**
  - Go to *Kernel > Change Kernel* in Jupyter
  - Or run:
    ```bash
    python -m ipykernel install --user --name=skybound-env --display-name "Python (SkyBound)"
    ```

---

## 📁 File Structure

```plaintext
.
├── main.py                        # CLI launcher
├── /src                          # Source scripts
│   ├── fetch_weather.py
│   ├── predict_launch.py
│   ├── train_model.py
│   └── ...
├── /models                       # Saved trained model
│   └── xgboost_best_model.pkl
├── SkyBound_Predictions.ipynb    # Forecast-based UI
├── predict_launch.ipynb          # Real-time/manual UI
├── requirements.txt
└── README.md
```

---

## 📜 License

MIT License – Free to use and modify.

---

## 👨‍🚀 Author

**Graham Cockerham**\
Bachelor of Science in Computer Science\
Western Governors University

---

## 🛠 Contributing

Pull requests are welcome. Please open an issue for major changes or feature proposals.

