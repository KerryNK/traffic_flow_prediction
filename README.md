# 🚗 Traffic Flow Prediction

## Overview

An AI-powered traffic volume prediction system using machine learning to help create more sustainable cities and communities.

### 🎯 Key Objectives

- Predict traffic patterns using historical data
- Reduce urban congestion
- Support smart city planning
- Contribute to SDG 11: Sustainable Cities

## 🛠️ Tech Stack

- Python 3.9+
- Scikit-learn
- Pandas
- TensorFlow (optional)
- Google Colab

## 📊 Dataset

Using Metro Interstate Traffic Volume dataset featuring:

- Weather conditions
- Time/Date information
- Holiday indicators
- Traffic volume measurements

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/KerryNK/traffic_flow_prediction
cd traffic_flow_prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook Traffic_Flow_Analysis.ipynb
```

## 📈 Model Architecture

We implement three approaches:

1. Linear Regression (baseline)
2. Random Forest (primary model)
3. Neural Network (experimental)

## 📋 Project Structure

``
traffic_flow_prediction/
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
│
├── notebooks/
│   └── Traffic_Flow_Analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
│
├── tests/
│   └── test_model.py
│
├── README.md
└── requirements.txt
``

## 🔍 Features

- Real-time traffic prediction
- Weather impact analysis
- Holiday traffic patterns
- Time-based volume forecasting

## 📝 Usage Example

```python
from src.model import TrafficPredictor

# Initialize predictor
predictor = TrafficPredictor()

# Make prediction
features = {
    'temp': 25.0,
    'rain_1h': 0.0,
    'clouds_all': 20,
    'hour': 14,
    'day': 2
}
predicted_volume = predictor.predict(features)
```

## 📊 Results

- MAE: 450 vehicles/hour
- R² Score: 0.85
- Prediction time: <100ms

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📫 Contact

Project Link: [https://github.com/KerryNK/traffic_flow_prediction](https://github.com/KerryNK/traffic_flow_prediction)

## 📜 License

MIT License © 2025 Kerry🧸💕
Project built for the PLP Academy

## SDGAssignment