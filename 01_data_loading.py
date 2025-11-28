"""
Data Loading and Preprocessing Module
Handles data acquisition and initial processing
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import os

class DataLoader:
    """Load time series data from various sources"""
    
    def __init__(self, config_path='../7_Config/config.yaml'):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.config = self._load_config(config_path)
    
    def load_stock_data(self, ticker='AAPL', periods=1000):
        """
        Load stock market data using yfinance
        
        Args:
            ticker: Stock symbol (default: AAPL)
            periods: Number of observations (minimum 500 as per requirements)
        
        Returns:
            pd.DataFrame: Time series data with features
        """
        print(f"Fetching {ticker} stock data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Calculate technical indicators
        data['Returns'] = data['Close'].pct_change()
        data['MA_7'] = data['Close'].rolling(window=7).mean()
        data['MA_30'] = data['Close'].rolling(window=30).mean()
        data['Volatility'] = data['Returns'].rolling(window=10).std()
        
        # Drop NaN values
        data = data.dropna()
        
        print(f"✓ Loaded {len(data)} observations of {ticker}")
        return data
    
    def load_energy_data(self, n_samples=1000):
        """
        Generate or load simulated energy load data
        
        Args:
            n_samples: Number of observations
        
        Returns:
            pd.DataFrame: Simulated time series data
        """
        print(f"Generating simulated energy load data ({n_samples} observations)...")
        
        # Simulate energy load with seasonality and trend
        t = np.arange(n_samples)
        seasonal = 50 * np.sin(2 * np.pi * t / 24) + 30 * np.cos(2 * np.pi * t / 168)
        trend = 0.1 * t
        noise = np.random.normal(0, 5, n_samples)
        
        energy_load = 100 + trend + seasonal + noise
        
        data = pd.DataFrame({
            'Timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H'),
            'Energy_Load': energy_load,
            'Hour': t % 24,
            'Day_of_Week': (t // 24) % 7,
            'Temperature': 20 + 15 * np.sin(2 * np.pi * (t % 8760) / 365) + np.random.normal(0, 2, n_samples)
        })
        
        print(f"✓ Generated {len(data)} observations of energy load")
        return data
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {'lookback': 30, 'forecast_horizon': 7}
    
    def create_sequences(self, data, lookback=30, forecast_horizon=7):
        """
        Create sequences for supervised learning
        
        Args:
            data: 1D array of time series values
            lookback: Number of timesteps to look back
            forecast_horizon: Number of steps to forecast
        
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_data_splits(self, data, feature_cols, lookback=30, 
                           forecast_horizon=7, val_split=0.1, test_split=0.1):
        """
        Prepare train, validation, and test splits
        
        Args:
            data: Input DataFrame
            feature_cols: Columns to use as features
            lookback: Number of timesteps
            forecast_horizon: Forecast steps
            val_split: Validation split ratio
            test_split: Test split ratio
        
        Returns:
            dict: Dictionary with processed data splits
        """
        print("Preparing data splits...")
        
        # Normalize features
        normalized_data = self.scaler.fit_transform(data[feature_cols])
        
        # Create sequences
        X, y = self.create_sequences(normalized_data.flatten() if len(feature_cols) == 1 
                                     else normalized_data[:, 0], 
                                     lookback, forecast_horizon)
        
        # Split into train, val, test (time series splits)
        n_total = len(X)
        n_train = int(n_total * (1 - val_split - test_split))
        n_val = int(n_total * val_split)
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scaler': self.scaler
        }
    
    def save_splits(self, splits, output_dir='../1_Data/data_splits'):
        """Save data splits to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'train_data.npy'), 
                np.concatenate([splits['X_train'], splits['y_train']], axis=1))
        np.save(os.path.join(output_dir, 'val_data.npy'), 
                np.concatenate([splits['X_val'], splits['y_val']], axis=1))
        np.save(os.path.join(output_dir, 'test_data.npy'), 
                np.concatenate([splits['X_test'], splits['y_test']], axis=1))
        
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(splits['scaler'], f)
        
        print(f"✓ Data splits saved to {output_dir}")


if __name__ == "__main__":
    loader = DataLoader()
    
    # Load stock data
    stock_data = loader.load_stock_data('AAPL', periods=1000)
    stock_data.to_csv('../1_Data/raw_stock_data.csv', index=True)
    
    # Prepare splits
    splits = loader.prepare_data_splits(stock_data, ['Close'], 
                                        lookback=30, forecast_horizon=7)
    loader.save_splits(splits)