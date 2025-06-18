import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
from pathlib import Path
from datetime import datetime
import logging

class TrafficPredictor:
    def __init__(self, model_name="RandomForest"):
        self.model = None
        self.features = [
            'temp', 'rain_1h', 'clouds_all', 'hour', 'day',
            'month', 'weekday', 'is_holiday'
        ]
        self.model_name = model_name
        self.setup_logging()
        self.setup_plotting()

    def setup_logging(self):
        logging.basicConfig(
            filename=f'logs/traffic_prediction_{datetime.now():%Y%m%d}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        Path('logs').mkdir(exist_ok=True)

    def setup_plotting(self):
        plt.style.use('seaborn')
        sns.set_context('notebook', font_scale=1.2)
        self.colors = sns.color_palette('husl', 10)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Enhanced data loading with feature engineering"""
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Time features
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['weekday'] = df['date_time'].dt.weekday
        
        # Holiday feature
        df['is_holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)
        
        # Weather normalization
        df['temp'] = (df['temp'] - df['temp'].mean()) / df['temp'].std()
        
        logging.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def train_model(self, df: pd.DataFrame) -> tuple:
        """Train with cross-validation and feature importance"""
        logging.info("Starting model training")
        X = df[self.features]
        y = df['traffic_volume']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        logging.info(f"Cross-validation scores: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Metrics
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        logging.info(f"Model metrics: {metrics}")
        return X_test, y_test, y_pred, metrics

    def plot_results(self, y_true, y_pred, metrics):
        """Enhanced visualizations with metrics"""
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)

        # 1. Predictions vs Actual
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(y_true[:100], label='Actual', color=self.colors[0], lw=2)
        ax1.plot(y_pred[:100], label='Predicted', color=self.colors[1], lw=2, ls='--')
        ax1.fill_between(range(100), y_true[:100], y_pred[:100], alpha=0.2, color=self.colors[1])
        ax1.set_title('Traffic Volume: Actual vs Predicted')
        ax1.legend()

        # 2. Feature Importance
        ax2 = fig.add_subplot(gs[1, 0])
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        sns.barplot(data=importance, y='feature', x='importance', ax=ax2, palette='husl')
        ax2.set_title('Feature Importance')

        # 3. Error Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        errors = y_true - y_pred
        sns.histplot(errors, kde=True, ax=ax3, color=self.colors[2])
        ax3.set_title('Prediction Error Distribution')

        # 4. Metrics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        metrics_text = '\n'.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        ax4.text(0.5, 0.5, f"Model Performance Metrics:\n\n{metrics_text}",
                ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('outputs/model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Create output directories
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Initialize and run
    predictor = TrafficPredictor()
    df = predictor.load_data('data/Metro_Interstate_Traffic_Volume.csv')
    X_test, y_test, y_pred, metrics = predictor.train_model(df)
    predictor.plot_results(y_test, y_pred, metrics)

if __name__ == "__main__":
    main()