import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.features = [
            'temp', 'rain_1h', 'clouds_all', 'hour', 'day',
            'month', 'weekday', 'is_holiday'
        ]

    @st.cache_data
    def load_data(self, file):
        df = pd.read_csv(file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['weekday'] = df['date_time'].dt.weekday
        df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)
        return df

    @st.cache_resource
    def train_model(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.model.fit(X, y)
        return self.model

    def load_model(self, model_path='models/traffic_model.pkl'):
        self.model = joblib.load(model_path)
        return self.model

    def predict(self, features):
        if not self.model:
            raise ValueError("Model not loaded or trained.")
        return self.model.predict([features])[0]

def main():
    st.set_page_config(page_title="Traffic Flow Prediction", layout="wide")
    st.title("🚗 Traffic Flow Prediction System")

    predictor = TrafficPredictor()

    # Sidebar
    st.sidebar.header("Model Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Traffic Data (CSV)", type="csv")

    if uploaded_file:
        df = predictor.load_data(uploaded_file)
        st.sidebar.success("Data loaded successfully!")

        # Data overview
        st.header("📊 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head())
        with col2:
            st.write("Dataset Shape:", df.shape)

        # Safe correlation heatmap toggle
        if st.checkbox("📌 Show Correlation Heatmap"):
            st.subheader("Feature Correlation Matrix")
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig_corr)

        # Model training
        X = df[predictor.features]
        y = df['traffic_volume']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                model = predictor.train_model(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                col1, col2 = st.columns(2)
                col1.metric("📊 Mean Absolute Error (MAE)", f"{mae:.2f}")
                col2.metric("📈 R² Score", f"{r2:.2f}")

                # Visualizations
                st.header("📈 Model Performance")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(y=y_test[:100], name="Actual"))
                fig1.add_trace(go.Scatter(y=y_pred[:100], name="Predicted"))
                fig1.update_layout(title="Traffic Volume: Actual vs Predicted")
                st.plotly_chart(fig1)

                importance = pd.DataFrame({
                    'feature': predictor.features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)

                fig2 = px.bar(importance, x='importance', y='feature',
                             title="Feature Importance")
                st.plotly_chart(fig2)

                joblib.dump(model, 'models/traffic_model.pkl')
                st.sidebar.success("Model trained and saved successfully!")

        # Prediction section
        st.header("🎯 Make Predictions")
        col1, col2 = st.columns(2)

        with col1:
            temp = st.slider("🌡 Temperature (°C)", -20.0, 40.0, 20.0)
            rain = st.slider("🌧 Rain (mm)", 0.0, 50.0, 0.0)
            clouds = st.slider("☁️ Cloud Coverage (%)", 0, 100, 50)
            hour = st.slider("🕓 Hour", 0, 23, 12)

        with col2:
            day = st.slider("📅 Day", 1, 31, 15)
            month = st.slider("📆 Month", 1, 12, 6)
            weekday = st.slider("🗓 Weekday", 0, 6, 3)
            is_holiday = st.checkbox("🎉 Is Holiday")

        if st.button("Predict"):
            try:
                predictor.load_model()
                features = [temp, rain, clouds, hour, day, month, weekday, int(is_holiday)]
                prediction = predictor.predict(features)
                st.success(f"Predicted Traffic Volume: {prediction:.0f} vehicles/hour")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.info("Please upload a traffic data CSV file to begin.")

if __name__ == "__main__":
    main()
