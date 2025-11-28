# Complete Usage Guide and Examples

## Quick Start (5 Minutes)

```python
# main_example.py
from main_pipeline import TimeSeriesPipeline

# Initialize pipeline with default configuration
pipeline = TimeSeriesPipeline(config_path='7_Config/config.yaml')

# Execute complete pipeline
pipeline.run_full_pipeline()

# Results will be saved to 4_Results/, 3_Models/, and 5_Report/
```

Run with:
```bash
python 2_Code/main_pipeline.py
```

---

## Detailed Usage Examples

### Example 1: Data Loading and Preprocessing

```python
import numpy as np
import pandas as pd
from data_loading import DataLoader

# Initialize data loader
loader = DataLoader(config_path='../7_Config/config.yaml')

# Option A: Load real stock data
print("Loading Apple stock data...")
stock_data = loader.load_stock_data(ticker='AAPL', periods=1000)
print(f"Loaded {len(stock_data)} observations")
print(stock_data.head())

# Option B: Generate simulated energy data
print("\nGenerating energy load data...")
energy_data = loader.load_energy_data(n_samples=1000)
print(energy_data.head())

# Prepare data splits (train, validation, test)
splits = loader.prepare_data_splits(
    data=stock_data,
    feature_cols=['Close'],
    lookback=30,
    forecast_horizon=7,
    val_split=0.1,
    test_split=0.2
)

# Save splits to disk
loader.save_splits(splits, output_dir='../1_Data/data_splits')

print(f"Train shape: {splits['X_train'].shape}")
print(f"Val shape: {splits['X_val'].shape}")
print(f"Test shape: {splits['X_test'].shape}")
```

### Example 2: Training Baseline Models

```python
import numpy as np
from baseline_model import BaselineModel, ARIMABaseline, evaluate_model

# Load preprocessed data (from Example 1)
# Assume splits dictionary is available

# Train Baseline LSTM
print("Training Baseline LSTM...")
baseline_lstm = BaselineModel(lookback=30, forecast_horizon=7)
history = baseline_lstm.train(
    X_train=splits['X_train'],
    y_train=splits['y_train'],
    X_val=splits['X_val'],
    y_val=splits['y_val'],
    epochs=100,
    batch_size=32
)

# Make predictions
y_pred_lstm = baseline_lstm.predict(splits['X_test'])

# Evaluate
metrics_lstm = evaluate_model(splits['y_test'][:, 0], y_pred_lstm[:, 0])
print("Baseline LSTM Metrics:")
for metric, value in metrics_lstm.items():
    print(f"  {metric}: {value:.6f}")

# Save model
baseline_lstm.save(model_path='../3_Models/baseline_lstm.h5')

# Train ARIMA Baseline
print("\nTraining ARIMA Baseline...")
arima = ARIMABaseline(order=(5, 1, 2))
arima.fit(splits['X_train'].flatten())
y_pred_arima = arima.forecast(steps=len(splits['X_test']))
print("✓ ARIMA model trained")
```

### Example 3: Training Attention Model

```python
import numpy as np
from attention_lstm_model import AttentionLSTMModel

# Initialize Attention LSTM
print("Building Seq2Seq Attention Model...")
attention_model = AttentionLSTMModel(lookback=30, forecast_horizon=7)

# Build architecture with optimal hyperparameters
attention_model.build_seq2seq_attention(
    lstm_units=128,
    attention_units=128,
    dropout_rate=0.2
)

# Train model
print("Training Attention LSTM...")
history = attention_model.train(
    X_train=splits['X_train'],
    y_train=splits['y_train'],
    X_val=splits['X_val'],
    y_val=splits['y_val'],
    epochs=100,
    batch_size=32
)

# Make predictions with attention weights
y_pred_attention, attention_weights = attention_model.predict(
    X_test=splits['X_test'],
    return_attention=True
)

print(f"Predictions shape: {y_pred_attention.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Evaluate model
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(splits['y_test'], y_pred_attention))
mae = mean_absolute_error(splits['y_test'], y_pred_attention)
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Save model
attention_model.save(model_path='../3_Models/attention_lstm.h5')

# Plot training history
attention_model.plot_training_history(
    save_path='../4_Results/visualizations/train_val_loss.png'
)
```

### Example 4: Hyperparameter Tuning

```python
import json
from hyperparameter_tuning import HyperparameterTuner
from attention_lstm_model import AttentionLSTMModel

# Define parameter grid
param_grid = {
    'lstm_units': [64, 128, 256],
    'dropout_rate': [0.1, 0.2, 0.3],
    'attention_units': [64, 128]
}

# Define model builder function
def build_attention_model(**params):
    model = AttentionLSTMModel(lookback=30, forecast_horizon=7)
    model.build_seq2seq_attention(**params)
    return model.model

# Initialize tuner
tuner = HyperparameterTuner(
    model_builder=build_attention_model,
    X_train=splits['X_train'],
    y_train=splits['y_train'],
    X_val=splits['X_val'],
    y_val=splits['y_val']
)

# Run grid search
print("Running Grid Search...")
results_df = tuner.grid_search(
    param_grid=param_grid,
    epochs=50,
    batch_size=32,
    verbose=True
)

# Save results
results_df.to_csv('../7_Config/grid_search_results.csv', index=False)

# Display best parameters
print("\nBest Parameters Found:")
print(json.dumps(tuner.best_params, indent=2))

# Summary statistics
summary = tuner.get_summary()
print("\nTuning Summary:")
for key, value in summary.items():
    if key != 'best_params':
        print(f"  {key}: {value}")
```

### Example 5: Rolling Origin Evaluation

```python
from hyperparameter_tuning import RollingOriginEvaluation
import pandas as pd

# Use the trained attention model from Example 3
# rolling_eval = RollingOriginEvaluation(...)

test_data_flat = splits['y_test'].flatten()

rolling_eval = RollingOriginEvaluation(
    model=attention_model,
    data=test_data_flat,
    initial_train_size=0.6,
    test_size=50
)

# Perform rolling evaluation
print("Executing Rolling Origin Evaluation...")
rolling_results = rolling_eval.evaluate(step_size=10)

# Display results
print("\nRolling Origin Evaluation Results:")
print(rolling_results)

# Save results
rolling_results.to_csv(
    '../4_Results/rolling_evaluation_results.csv',
    index=False
)

# Summary statistics
print("\nMetrics Summary:")
print(rolling_results[['rmse', 'mae', 'mape']].describe())
```

### Example 6: Attention Analysis and Visualization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract attention weights from predictions
attention_weights = attention_model.attention_weights

print(f"Attention weights shape: {attention_weights.shape}")

# Aggregate attention across test set
avg_attention = np.mean(attention_weights, axis=0)

# Find most important timesteps
print("\nAverage Attention Distribution:")
print(f"  Max: {avg_attention.max():.4f}")
print(f"  Min: {avg_attention.min():.4f}")
print(f"  Mean: {avg_attention.mean():.4f}")

# Identify top attended timesteps
top_k = 5
top_indices = np.argsort(avg_attention)[-top_k:][::-1]

print(f"\nTop {top_k} Most Attended Timesteps:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. Timestep {idx}: {avg_attention[idx]:.4f}")

# Save attention analysis
attention_df = pd.DataFrame({
    'timestep': np.arange(len(avg_attention)),
    'avg_attention_weight': avg_attention
})
attention_df.to_csv(
    '../4_Results/attention_weights_analysis.csv',
    index=False
)

# Visualize attention heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(attention_weights[:50], cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'})
plt.title('Attention Weights Heatmap (First 50 Test Samples)', fontsize=14)
plt.xlabel('Historical Timesteps (Lookback Window)')
plt.ylabel('Test Sample')
plt.tight_layout()
plt.savefig('../4_Results/visualizations/attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot attention distribution
plt.figure(figsize=(12, 5))
plt.plot(avg_attention, marker='o', linewidth=2, markersize=6)
plt.fill_between(range(len(avg_attention)), avg_attention, alpha=0.3)
plt.title('Average Attention Distribution Across Lookback Window', fontsize=14)
plt.xlabel('Historical Timesteps')
plt.ylabel('Average Attention Weight')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../4_Results/visualizations/attention_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Attention visualizations saved")
```

### Example 7: Complete Comparison Analysis

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Assuming predictions from all models are available:
# y_pred_lstm, y_pred_arima, y_pred_attention

# Calculate metrics for all models
def calculate_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

y_test_flat = splits['y_test'][:, 0]

# Calculate metrics
results = []
results.append(calculate_metrics(y_test_flat, y_pred_lstm[:, 0], 'LSTM Baseline'))
results.append(calculate_metrics(y_test_flat, y_pred_arima[:len(y_test_flat)], 'ARIMA'))
results.append(calculate_metrics(y_test_flat, y_pred_attention[:, 0], 'Seq2Seq+Attention'))

comparison_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Calculate improvement
lstm_rmse = comparison_df[comparison_df['Model'] == 'LSTM Baseline']['RMSE'].values[0]
attention_rmse = comparison_df[comparison_df['Model'] == 'Seq2Seq+Attention']['RMSE'].values[0]
improvement = ((lstm_rmse - attention_rmse) / lstm_rmse) * 100

print(f"\n✓ Attention model improvement over LSTM: {improvement:.1f}%")

# Save comparison
comparison_df.to_csv('../4_Results/model_comparison.csv', index=False)
```

### Example 8: Loading Trained Models for Inference

```python
import tensorflow as tf
import numpy as np
from data_loading import DataLoader
import pickle

# Load scaler
with open('../1_Data/data_splits/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load trained Attention model
attention_model = tf.keras.models.load_model('../3_Models/attention_lstm.h5')

# Load new data
loader = DataLoader()
new_stock_data = loader.load_stock_data('AAPL', periods=100)

# Prepare for prediction (last 30 observations)
last_30 = new_stock_data['Close'].values[-30:].reshape(-1, 1)
last_30_scaled = scaler.transform(last_30)

# Make prediction
X_test = last_30_scaled.flatten().reshape(1, 30)
prediction = attention_model.predict([X_test, np.zeros((1, 7, 1))])

# Inverse transform to original scale
prediction_original = scaler.inverse_transform(prediction.reshape(-1, 1))

print("7-Day Forecast:")
for i in range(7):
    print(f"  Day {i+1}: ${prediction_original[i, 0]:.2f}")
```

---

## Configuration Customization

### Modify config.yaml

```yaml
# Example: Use energy data instead of stock data
data:
  source: 'energy'  # Changed from 'stock'
  periods: 1000
  lookback: 24      # Changed for hourly data (1 day)
  forecast_horizon: 24  # Forecast 1 day ahead

# Reduce model size for faster training
model:
  lstm_units: 64    # Reduced from 128
  dropout_rate: 0.1
  attention_units: 64

# Fewer epochs for quick testing
training:
  epochs: 20        # Reduced from 100
  batch_size: 16
```

---

## Common Tasks

### Task: Train only baseline for quick comparison
```python
from baseline_model import BaselineModel

baseline = BaselineModel()
baseline.train(splits['X_train'], splits['y_train'],
               splits['X_val'], splits['y_val'], epochs=50)
baseline.save()
```

### Task: Extract predictions for export
```python
predictions_dict = {
    'lstm': y_pred_lstm,
    'attention': y_pred_attention,
    'actual': splits['y_test']
}

import json
np.save('../4_Results/predictions.npy', predictions_dict)
```

### Task: Batch prediction on new data
```python
# Prepare multiple sequences
new_data = np.random.randn(100, 30)  # 100 sequences, 30 timesteps each
new_decoder_input = np.zeros((100, 7, 1))

# Predict
batch_predictions = attention_model.predict(
    [new_data.reshape(100, 30, 1), new_decoder_input]
)
print(f"Batch predictions shape: {batch_predictions.shape}")
```

---

## Troubleshooting

### Issue: "OutOfMemory" error
```python
# Reduce batch size in config.yaml
batch_size: 8  # Reduced from 32
```

### Issue: Model not converging
```python
# Adjust learning rate
training:
  learning_rate: 0.0005  # Reduced learning rate
  epochs: 200           # More epochs
```

### Issue: Slow training on CPU
```python
# Use Google Colab with GPU
# Or specify GPU in code:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

---

For more information, see:
- README.md - Project overview
- SETUP.md - Installation guide  
- PROJECT_REPORT.md - Technical details