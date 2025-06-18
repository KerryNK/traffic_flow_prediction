import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore

# ------------------------------------
# Traffic Predictor Class
# ------------------------------------
class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.features = ['temp', 'rain_1h', 'clouds_all', 'hour', 'day']

    def load_model(self, path='models/traffic_model.pkl'):
        """Load a trained model from disk"""
        self.model = joblib.load(path)
        print(f"📥 Model loaded from {path}")

    def predict(self, feature_list: list) -> float:
        """Predict traffic volume"""
        if not self.model:
            raise ValueError("Model not loaded.")
        return self.model.predict([feature_list])[0]

# ------------------------------------
# Streamlit App
# ------------------------------------
st.set_page_config(page_title="Traffic Volume Predictor", layout="centered")
st.title("🚦 Traffic Volume Predictor")
st.markdown("Predict hourly traffic volume based on weather and time data. Built for **SDG 11: Sustainable Cities**.")

# Load model
predictor = TrafficPredictor()
predictor.load_model()

# UI Inputs
temp = st.slider("🌡 Temperature (°C)", -10.0, 40.0, 25.0)
rain_1h = st.slider("🌧 Rainfall (mm)", 0.0, 50.0, 0.0)
clouds_all = st.slider("☁️ Cloud Coverage (%)", 0, 100, 20)
hour = st.slider("🕓 Hour of Day", 0, 23, 14)
day_name = st.selectbox("📅 Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Map day to int
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
day = day_map[day_name]

# Predict
if st.button("Predict Traffic Volume"):
    features = [temp, rain_1h, clouds_all, hour, day]
    result = predictor.predict(features)
    st.success(f"🚗 Predicted Traffic Volume: **{int(result)} vehicles/hour**")
