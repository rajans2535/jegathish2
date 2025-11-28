"""
Baseline Model Implementation (LSTM without Attention)
Implements standard LSTM and ARIMA for comparison
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import joblib

class BaselineModel:
    """Standard LSTM baseline without attention"""
    
    def __init__(self, lookback=30, forecast_horizon=7):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.history = None
    
    def build_lstm_baseline(self, input_shape, lstm_units=128, dropout_rate=0.2):
        """
        Build standard LSTM model
        
        Args:
            input_shape: Shape of input sequences (lookback,)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate to prevent overfitting
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        print("✓ Baseline LSTM model built")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train baseline model"""
        self.model = self.build_lstm_baseline((X_train.shape[1], 1))
        
        # Reshape for LSTM
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        self.history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=epochs, batch_size=batch_size,
            verbose=1, patience=10
        )
        
        print("✓ Baseline LSTM training completed")
        return self.history
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predictions = self.model.predict(X_test_reshaped)
        return predictions
    
    def save(self, model_path='../3_Models/baseline_lstm.h5'):
        """Save model"""
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")


class ARIMABaseline:
    """ARIMA baseline for comparison"""
    
    def __init__(self, order=(5, 1, 2)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data):
        """Fit ARIMA model to training data"""
        self.fitted_model = ARIMA(train_data, order=self.order).fit()
        print(f"✓ ARIMA{self.order} model fitted")
        return self.fitted_model
    
    def forecast(self, steps):
        """Generate forecast for specified steps"""
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        return forecast_result.predicted_mean.values


def plot_baseline_results(y_actual, y_pred, save_path='../4_Results/visualizations/predictions_plot.png'):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(14, 5))
    
    plt.plot(y_actual, label='Actual', linewidth=2, color='blue')
    plt.plot(y_pred, label='Predicted (Baseline LSTM)', linewidth=2, color='orange', alpha=0.7)
    
    plt.title('Baseline Model: Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Normalized Values', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()


def evaluate_model(y_actual, y_pred):
    """
    Calculate evaluation metrics
    
    Returns:
        dict: Dictionary with all metrics
    """
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return metrics


if __name__ == "__main__":
    from data_loading import DataLoader
    
    # Load data
    loader = DataLoader()
    stock_data = loader.load_stock_data('AAPL', periods=1000)
    splits = loader.prepare_data_splits(stock_data, ['Close'], 
                                        lookback=30, forecast_horizon=7)
    
    # Train baseline
    baseline = BaselineModel()
    baseline.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        epochs=50
    )
    
    # Evaluate
    y_pred = baseline.predict(splits['X_test'])
    metrics = evaluate_model(splits['y_test'][:, 0], y_pred[:, 0])
    print("Baseline LSTM Metrics:", metrics)