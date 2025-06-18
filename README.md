# 🚗 Traffic Flow Prediction

## 🌍 Project for SDG 11: Sustainable Cities and Communities

This project is part of the **PLP Academy Week 2 Assignment – AI for Sustainable Development**. It directly addresses **UN Sustainable Development Goal 11** by using machine learning to analyze and predict urban traffic flow, aiming to reduce congestion, support smart transport planning, and improve quality of life in cities.

---

## 🎯 Key Objectives

- Predict traffic patterns using historical data
- Reduce urban congestion and emissions
- Support sustainable urban mobility
- Inform smart city infrastructure decisions

---

## 🛠️ Tech Stack

- Python 3.9+
- Scikit-learn
- Pandas
- TensorFlow (optional for deep learning model)
- Google Colab / Jupyter Notebook

---

## 📊 Dataset

Using the [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) which contains:

- Hourly traffic volume measurements
- Weather conditions (temp, rain, snow, clouds)
- Time-based indicators (hour, weekday, holiday)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/KerryNK/traffic_flow_prediction
cd traffic_flow_prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook notebooks/Traffic_Flow_Analysis.ipynb

📈 Model Architecture
Three ML models were tested and compared:

✅ Linear Regression – as a simple baseline

✅ Random Forest Regressor – primary production model

✅ Neural Network – experimental model using TensorFlow

📋 Project Structure

traffic_flow_prediction/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── notebooks/
│   └── Traffic_Flow_Analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── tests/
│   └── test_model.py
├── README.md
└── requirements.txt

🔍 Features
⏱️ Real-time hourly traffic volume prediction

🌧️ Weather impact forecasting

📅 Holiday traffic variation detection

🕓 Hourly & daily congestion pattern insights

🧪 Usage Example
from src.model import TrafficPredictor

# Initialize predictor
predictor = TrafficPredictor()

# Example input features
features = {
    'temp': 25.0,
    'rain_1h': 0.0,
    'clouds_all': 20,
    'hour': 14,
    'day': 2
}

# Make a traffic volume prediction
predicted_volume = predictor.predict(features)

📊 Results
🔍 MAE: 450 vehicles/hour

🎯 R² Score: 0.85

⚡ Average prediction time: <100ms

🧠 Ethical & Social Impact
✅ Inclusivity: The model can help identify high-congestion areas in under-served neighborhoods to improve transit access equitably.

⚠️ Bias Awareness: Traffic data may not represent informal or pedestrian transit—future iterations should include more diverse mobility data.

🌱 Sustainability: By forecasting peak traffic, city planners can reduce carbon emissions by rerouting or adjusting transport frequency.

🤝 Community Benefit: Can be shared with local governments or mobility apps to improve urban planning for everyone.

🎓 Contribution Guide
Fork the repository

Create a new branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a pull request ✅

📫 Contact
Project Repo: https://github.com/KerryNK/traffic_flow_prediction

![GitHub repo size](https://img.shields.io/github/repo-size/KerryNK/traffic_flow_prediction)

![MIT license](https://img.shields.io/github/license/KerryNK/traffic_flow_prediction)

Made with ❤️ by Kerry🧸💕 for the PLP Academy AI for SDG Challenge
