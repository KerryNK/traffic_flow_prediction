import pickle
import pandas as pd

def load_model():
    with open('../models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict_traffic(features):
    model = load_model()
    prediction = model.predict([features])[0]
    return round(prediction)

def main():
    # Example prediction
    sample_features = {
        'hour': 8,
        'day': 15,
        'month': 6,
        'temp': 25.0,
        'rain_1h': 0.0,
        'snow_1h': 0.0,
        'clouds_all': 20
    }

    traffic = predict_traffic(list(sample_features.values()))
    print(f"Predicted traffic volume: {traffic} vehicles/hour")

if __name__ == "__main__":
    main()