"""
Hyperparameter Tuning Module (Task 3)
Implements Grid Search for optimal hyperparameter selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import itertools
from datetime import datetime
import json
import os

class HyperparameterTuner:
    """Grid search for optimal hyperparameters"""
    
    def __init__(self, model_builder, X_train, y_train, X_val, y_val):
        """
        Args:
            model_builder: Function that builds and returns a model
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        self.model_builder = model_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
    
    def grid_search(self, param_grid, epochs=50, batch_size=32, verbose=True):
        """
        Perform grid search over hyperparameter space
        
        Args:
            param_grid: Dictionary with parameter names and lists of values
                Example: {
                    'lstm_units': [64, 128, 256],
                    'dropout_rate': [0.1, 0.2, 0.3],
                    'attention_units': [64, 128]
                }
            epochs: Training epochs for each model
            batch_size: Batch size for training
            verbose: Print progress
        
        Returns:
            pd.DataFrame: Results for all parameter combinations
        """
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        print(f"Starting Grid Search with {total_combinations} combinations...")
        print("=" * 80)
        
        for idx, params in enumerate(param_combinations, 1):
            if verbose:
                print(f"\n[{idx}/{total_combinations}] Testing parameters: {params}")
            
            try:
                # Build model with current parameters
                model = self.model_builder(**params)
                
                # Train model
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[
                        self._get_early_stopping_callback()
                    ]
                )
                
                # Evaluate
                val_loss = history.history['val_loss'][-1]
                val_mae = history.history['val_mae'][-1]
                
                # Store results
                result = {
                    'params': params,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'epochs_trained': len(history.history['loss']),
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                
                # Track best
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params
                
                if verbose:
                    print(f"  → Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
            
            except Exception as e:
                print(f"  ✗ Error with parameters {params}: {str(e)}")
                continue
        
        print("\n" + "=" * 80)
        print(f"Grid Search Complete!")
        print(f"Best Parameters: {self.best_params}")
        print(f"Best Validation Loss: {self.best_score:.6f}")
        
        return pd.DataFrame(self.results)
    
    def random_search(self, param_distributions, n_iter=20, epochs=50, verbose=True):
        """
        Perform random search over hyperparameter space
        
        Args:
            param_distributions: Dictionary with parameter names and distributions
            n_iter: Number of random combinations to try
            epochs: Training epochs for each model
            verbose: Print progress
        
        Returns:
            pd.DataFrame: Results for all sampled combinations
        """
        print(f"Starting Random Search with {n_iter} iterations...")
        print("=" * 80)
        
        for idx in range(n_iter):
            # Sample random parameters
            params = {
                key: np.random.choice(values)
                for key, values in param_distributions.items()
            }
            
            if verbose:
                print(f"\n[{idx+1}/{n_iter}] Testing parameters: {params}")
            
            try:
                model = self.model_builder(**params)
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=32,
                    verbose=0,
                    callbacks=[self._get_early_stopping_callback()]
                )
                
                val_loss = history.history['val_loss'][-1]
                val_mae = history.history['val_mae'][-1]
                
                result = {
                    'params': params,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'epochs_trained': len(history.history['loss']),
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params
                
                if verbose:
                    print(f"  → Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
            
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue
        
        print("\n" + "=" * 80)
        print(f"Random Search Complete!")
        print(f"Best Parameters: {self.best_params}")
        print(f"Best Validation Loss: {self.best_score:.6f}")
        
        return pd.DataFrame(self.results)
    
    def _get_early_stopping_callback(self):
        """Get early stopping callback"""
        from tensorflow.keras.callbacks import EarlyStopping
        return EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
    
    def save_results(self, output_path='../7_Config/hyperparameter_tuning_results.csv'):
        """Save tuning results to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_path, index=False)
        
        # Also save best params as JSON
        best_params_path = output_path.replace('.csv', '_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        print(f"✓ Best params saved to {best_params_path}")
        
        return results_df
    
    def get_summary(self):
        """Get summary statistics of tuning"""
        results_df = pd.DataFrame(self.results)
        
        summary = {
            'total_combinations': len(self.results),
            'best_val_loss': results_df['val_loss'].min(),
            'worst_val_loss': results_df['val_loss'].max(),
            'mean_val_loss': results_df['val_loss'].mean(),
            'std_val_loss': results_df['val_loss'].std(),
            'best_params': self.best_params
        }
        
        return summary


class RollingOriginEvaluation:
    """
    Implements rolling origin evaluation for time series
    (Task 4 - Rigorous Evaluation)
    """
    
    def __init__(self, model, data, initial_train_size=0.6, test_size=50):
        """
        Args:
            model: Trained model to evaluate
            data: Full time series data
            initial_train_size: Initial training set size (fraction)
            test_size: Size of each test window
        """
        self.model = model
        self.data = data
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.rolling_metrics = []
    
    def evaluate(self, step_size=10):
        """
        Perform rolling origin evaluation
        
        Args:
            step_size: Number of observations to advance in each iteration
        
        Returns:
            pd.DataFrame: Metrics for each rolling window
        """
        n = len(self.data)
        initial_size = int(n * self.initial_train_size)
        
        window_idx = 0
        for t in range(initial_size, n - self.test_size, step_size):
            window_idx += 1
            
            # Train on data up to time t
            train_data = self.data[:t]
            
            # Test on next test_size observations
            test_data = self.data[t:t+self.test_size]
            
            # Make predictions
            predictions = self.model.predict(train_data)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            mae = mean_absolute_error(test_data, predictions)
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            self.rolling_metrics.append({
                'window': window_idx,
                'train_end': t,
                'test_start': t,
                'test_end': t + self.test_size,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })
            
            print(f"Window {window_idx}: RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
        
        return pd.DataFrame(self.rolling_metrics)


if __name__ == "__main__":
    print("Hyperparameter Tuning Module")
    print("Use GridSearchCV or this tuner with your model")
    
    # Example parameter grid
    param_grid = {
        'lstm_units': [64, 128, 256],
        'dropout_rate': [0.1, 0.2, 0.3],
        'attention_units': [64, 128]
    }
    
    print(f"Example grid size: {len(list(ParameterGrid(param_grid)))} combinations")