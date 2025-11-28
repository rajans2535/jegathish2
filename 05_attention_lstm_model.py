"""
Attention LSTM Model Implementation (Seq2Seq with Self-Attention)
Core architecture for the advanced time series forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Dropout, Concatenate, 
    Dot, Softmax, RepeatVector, TimeDistributed, Layer
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionLayer(Layer):
    """
    Custom Attention Layer for Seq2Seq models
    Implements Bahdanau-style attention mechanism
    """
    
    def __init__(self, attention_units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        
        self.W1 = Dense(attention_units)
        self.W2 = Dense(attention_units)
        self.V = Dense(1)
    
    def call(self, query, values):
        """
        Compute attention weights
        
        Args:
            query: Decoder hidden state (batch_size, hidden_size)
            values: Encoder outputs (batch_size, seq_len, hidden_size)
        
        Returns:
            context: Weighted average of values (batch_size, hidden_size)
            attention_weights: Attention scores (batch_size, seq_len)
        """
        # Expand query for broadcasting
        query_expanded = tf.expand_dims(query, 1)  # (batch_size, 1, hidden_size)
        
        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_expanded)))
        score = tf.squeeze(score, -1)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch_size, seq_len)
        
        # Apply attention to values
        context = tf.reduce_sum(
            tf.expand_dims(attention_weights, -1) * values, axis=1
        )  # (batch_size, hidden_size)
        
        return context, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'attention_units': self.attention_units})
        return config


class AttentionLSTMModel:
    """Seq2Seq model with Attention mechanism"""
    
    def __init__(self, lookback=30, forecast_horizon=7):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.encoder = None
        self.decoder = None
        self.attention_layer = None
        self.history = None
        self.attention_weights = None
    
    def build_seq2seq_attention(self, lstm_units=128, attention_units=128, dropout_rate=0.2):
        """
        Build Seq2Seq model with Attention
        
        Architecture:
        - Encoder: Bidirectional LSTM with attention
        - Decoder: LSTM that attends over encoder outputs
        - Output: Dense layer for final prediction
        """
        
        # Encoder
        encoder_inputs = Input(shape=(self.lookback, 1), name='encoder_input')
        encoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True, 
                           dropout=dropout_rate, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        
        # Decoder
        decoder_inputs = Input(shape=(self.forecast_horizon, 1), name='decoder_input')
        decoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True,
                           dropout=dropout_rate, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
        
        # Attention mechanism
        self.attention_layer = AttentionLayer(attention_units=attention_units, name='attention')
        
        # Apply attention to each decoder output
        attention_outputs = []
        for i in range(self.forecast_horizon):
            context, _ = self.attention_layer(decoder_outputs[:, i, :], encoder_outputs)
            attention_outputs.append(context)
        
        # Stack attention outputs
        attention_outputs = tf.stack(attention_outputs, axis=1)  # (batch, forecast_horizon, lstm_units)
        
        # Final dense layer
        final_outputs = TimeDistributed(Dense(1), name='time_distributed_dense')(attention_outputs)
        
        # Reshape output
        final_outputs = tf.squeeze(final_outputs, -1)  # (batch, forecast_horizon)
        
        # Create model
        self.model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=final_outputs,
            name='seq2seq_attention'
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ Seq2Seq Attention model built successfully")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train Attention LSTM model
        
        Args:
            X_train: Training sequences (batch, lookback)
            y_train: Training targets (batch, forecast_horizon)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Reshape for LSTM input
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Decoder inputs (shifted y_train, starting with zeros)
        y_train_decoder = np.concatenate([
            np.zeros((X_train_reshaped.shape[0], 1, 1)),
            y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, :-1, :]
        ], axis=1)
        
        y_val_decoder = np.concatenate([
            np.zeros((X_val_reshaped.shape[0], 1, 1)),
            y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, :-1, :]
        ], axis=1)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            [X_train_reshaped, y_train_decoder],
            y_train,
            validation_data=([X_val_reshaped, y_val_decoder], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("✓ Attention LSTM training completed")
        return self.history
    
    def predict(self, X_test, return_attention=False):
        """
        Make predictions and optionally return attention weights
        
        Args:
            X_test: Test sequences
            return_attention: If True, also return attention weights
        
        Returns:
            predictions: Model predictions
            attention_weights (optional): Attention weight matrices
        """
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create decoder inputs for test
        y_test_decoder = np.zeros((X_test_reshaped.shape[0], self.forecast_horizon, 1))
        
        predictions = self.model.predict([X_test_reshaped, y_test_decoder])
        
        if return_attention:
            # Extract attention weights for analysis
            attention_weights = self._extract_attention_weights(X_test_reshaped, y_test_decoder)
            return predictions, attention_weights
        
        return predictions
    
    def _extract_attention_weights(self, X_test, y_decoder):
        """Extract attention weights for interpretability"""
        # Create intermediate model to extract attention outputs
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=self.attention_layer.output[1]  # Attention weights
        )
        
        weights = attention_model.predict([X_test, y_decoder])
        self.attention_weights = weights
        return weights
    
    def save(self, model_path='../3_Models/attention_lstm.h5'):
        """Save model"""
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    def plot_training_history(self, save_path='../4_Results/visualizations/train_val_loss.png'):
        """Plot training and validation loss"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss (MSE)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        plt.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        plt.title('Model MAE', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    from data_loading import DataLoader
    
    # Load data
    loader = DataLoader()
    stock_data = loader.load_stock_data('AAPL', periods=1000)
    splits = loader.prepare_data_splits(stock_data, ['Close'])
    
    # Build and train model
    attention_model = AttentionLSTMModel()
    attention_model.build_seq2seq_attention()
    attention_model.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        epochs=50
    )
    
    # Make predictions
    predictions, weights = attention_model.predict(splits['X_test'], return_attention=True)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Attention weights shape: {weights.shape}")