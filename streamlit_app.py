import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# --- Helper Functions ---

@st.cache_data
def load_sample_data():
    # Create a sample DataFrame resembling the original traffic data
    n = 500
    rng = pd.date_range("2022-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "date_time": rng,
        "temp": np.random.uniform(-10, 35, n),
        "rain_1h": np.random.uniform(0, 10, n),
        "clouds_all": np.random.randint(0, 100, n),
        "holiday": np.random.choice(['None', 'Holiday'], n, p=[0.85, 0.15]),
        "traffic_volume": np.random.randint(500, 7000, n),
    })
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)
    return df

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)
    return model

def plot_actual_vs_pred(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:100], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred[:100], mode='lines', name='Predicted'))
    fig.update_layout(title="Traffic Volume: Actual vs Predicted", xaxis_title="Sample", yaxis_title="Traffic Volume")
    st.plotly_chart(fig)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fig = go.Figure(go.Bar(
        x=importances,
        y=feature_names,
        orientation='h',
        marker=dict(color=importances, colorscale='Viridis')
    ))
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig)

# --- Streamlit UI ---

st.set_page_config(page_title="🚦 Traffic Flow Prediction", layout="wide")
st.title("🚦 Traffic Flow Prediction App")

st.write("Using sample traffic data (no upload required).")

df = load_sample_data()
st.success("Sample data loaded successfully!")
st.write("### Data Sample", df.head())

feature_cols = ['temp', 'rain_1h', 'clouds_all', 'hour', 'day', 'month', 'weekday', 'is_holiday']
X = df[feature_cols]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("Train Model"):
    with st.spinner("Training model..."):
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("Model trained!")

        st.write("### Model Performance")
        st.write(f"**R² Score:** {model.score(X_test, y_test):.3f}")
        st.write(f"**MAE:** {np.mean(np.abs(y_test - y_pred)):.2f}")

        plot_actual_vs_pred(y_test.values, y_pred)
        plot_feature_importance(model, feature_cols)

st.write("### Make a Prediction")
col1, col2 = st.columns(2)
with col1:
    temp = st.slider("Temperature (°C)", -20.0, 40.0, 20.0)
    rain = st.slider("Rain (mm)", 0.0, 50.0, 0.0)
    clouds = st.slider("Cloud Coverage (%)", 0, 100, 50)
    hour = st.slider("Hour", 0, 23, 12)
with col2:
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, 3)
    is_holiday = st.checkbox("Is Holiday", value=False)

if st.button("Predict Traffic Volume"):
    if 'model' not in locals():
        model = train_model(X_train, y_train)
    features = [temp, rain, clouds, hour, day, month, weekday, int(is_holiday)]
    prediction = model.predict([features])[0]
    st.success(f"Predicted Traffic Volume: {prediction:.0f} vehicles/hour")
