import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def load_data():
    df = pd.read_csv('../data/Metro_Interstate_Traffic_Volume.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

def prepare_features(df):
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    features = ['hour', 'day', 'month', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    X = df[features]
    y = df['traffic_volume']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def main():
    df = load_data()
    X, y = prepare_features(df)
    model, X_test, y_test = train_model(X, y)

    # Save model
    with open('../models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Print metrics
    score = model.score(X_test, y_test)
    print(f"Model R² Score: {score:.3f}")

if __name__ == "__main__":
    main()