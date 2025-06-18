import streamlit as st
import numpy as np
import joblib  # or pickle if you saved the model using pickle

# Load your trained model
# Update with the correct path to your saved model
# model = joblib.load('models/random_forest_model.pkl')

# Dummy model for demonstration purposes
def dummy_predict(features):
    # Example prediction logic (replace with real model.predict call)
    base_volume = 3000
    modifiers = features['temp'] * 2 - features['clouds_all'] + features['hour'] * 10
    return base_volume + modifiers

# Streamlit UI
st.set_page_config(page_title="Traffic Volume Predictor", layout="centered")
st.title("🚗 Traffic Volume Predictor")
st.markdown("Predict hourly traffic flow using weather and time data. Powered by ML and built for SDG 11 🌍")

# Inputs
temp = st.slider("Temperature (°C)", -10.0, 40.0, 25.0)
rain_1h = st.slider("Rainfall in the last hour (mm)", 0.0, 50.0, 0.0)
clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 20)
hour = st.slider("Hour of Day (24hr)", 0, 23, 14)
day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Convert day to numerical value
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
day_num = day_map[day]

# Predict button
if st.button("Predict Traffic Volume"):
    features = {
        'temp': temp,
        'rain_1h': rain_1h,
        'clouds_all': clouds_all,
        'hour': hour,
        'day': day_num
    }

    # prediction = model.predict([[temp, rain_1h, clouds_all, hour, day_num]])
    prediction = dummy_predict(features)

    st.success(f"🚦 Predicted Traffic Volume: {int(prediction)} vehicles/hour")
