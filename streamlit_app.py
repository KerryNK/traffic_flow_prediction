import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Set styling for plots
plt.style.use('seaborn')
sns.set_palette('husl')
colors = sns.color_palette('husl', 8)

class TrafficPredictor:
    def __init__(self, model_path='models/traffic_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = ['temp', 'rain_1h', 'clouds_all', 'hour', 'day']

    def train(self, data_path):
        """Train the traffic prediction model"""
        # Load and prepare data
        df = pd.read_csv(data_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day

        # Prepare features
        X = df[self.feature_names]
        y = df['traffic_volume']

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Visualize results
        self._plot_predictions(y_test, y_pred)
        self._plot_feature_importance()
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
        return r2_score(y_test, y_pred)

    def predict(self, features):
        """Make traffic prediction"""
        if not self.model:
            self.model = joblib.load(self.model_path)
        return self.model.predict([features])[0]

    def _plot_predictions(self, y_true, y_pred, samples=100):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:samples], label='Actual', color=colors[0], linewidth=2)
        plt.plot(y_pred[:samples], label='Predicted', color=colors[1], linewidth=2, linestyle='--')
        plt.fill_between(range(samples), y_true[:samples], y_pred[:samples], 
                        alpha=0.2, color=colors[1])
        plt.title('Traffic Volume: Actual vs Predicted', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Traffic Volume', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.show()

    def _plot_feature_importance(self):
        """Plot feature importance"""
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance['feature'], importance['importance'], 
                       color=colors[2:])
        plt.title('Feature Importance', fontsize=14)
        plt.xlabel('Importance Score', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        plt.show()

def main():
    predictor = TrafficPredictor()
    score = predictor.train('data/Metro_Interstate_Traffic_Volume.csv')
    print(f"\nModel R² Score: {score:.3f}")

    # Example prediction
    sample = [25.0, 0.0, 20, 8, 1]  # temp, rain, clouds, hour, day
    prediction = predictor.predict(sample)
    print(f"\nPredicted Traffic Volume: {prediction:.0f} vehicles/hour")

if __name__ == "__main__":
    main()