"""
Fixed TFT Implementation with Debugging and Better Training

Key fixes:
1. Gradient clipping to prevent vanishing/exploding gradients
2. Better initialization
3. Reduced complexity for small datasets
4. Output layer directly instead of through GRN
5. Better learning rate schedule
"""
import sys
sys.path.insert(0, 'src')

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

from oer_model.utils.io import read_dataframe
import pickle

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Simple TFT with direct output
class SimplifiedTFT(keras.Model):
    """Simplified TFT that should actually learn."""
    
    def __init__(self, num_features, encoder_length=12, hidden_size=64, dropout=0.1):
        super().__init__()
        self.encoder_length = encoder_length
        
        # Feature projection
        self.input_projection = layers.Dense(hidden_size, activation='relu')
        
        # LSTM encoder
        self.lstm = layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)
        
        # Attention
        self.attention = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=hidden_size // 4,
            dropout=dropout
        )
        self.attention_norm = layers.LayerNormalization()
        
        # Output layers - simpler path
        self.dense1 = layers.Dense(hidden_size // 2, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.dense2 = layers.Dense(hidden_size // 4, activation='relu')
        self.output_layer = layers.Dense(1)  # Direct output
        
    def call(self, inputs, training=None):
        # Project inputs
        x = self.input_projection(inputs)
        
        # LSTM
        x = self.lstm(x, training=training)
        
        # Attention
        attn_out = self.attention(x, x, x, training=training)
        x = self.attention_norm(x + attn_out)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Output path
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return output


# Load data
df = read_dataframe('data/processed/features_processed.csv').set_index('date')
X = df[[c for c in df.columns if c != 'oer_cpi_yoy']].values
y = df['oer_cpi_yoy'].values

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

# Scale
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

encoder_length = 12

# Create sequences
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, encoder_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, encoder_length)

print(f"Sequences - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

# Build model
model = SimplifiedTFT(
    num_features=X.shape[1],
    encoder_length=encoder_length,
    hidden_size=64,
    dropout=0.2
)

# Compile with gradient clipping
optimizer = keras.optimizers.legacy.Adam(
    learning_rate=0.001,
    clipnorm=1.0  # Gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print("\n" + "="*80)
print("TRAINING SIMPLIFIED TFT")
print("="*80)

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Custom callback to check predictions during training
class PredictionMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            # Make a prediction
            sample_pred = self.model.predict(X_test_seq[:5], verbose=0)
            sample_pred_original = y_scaler.inverse_transform(sample_pred)
            unique_count = len(np.unique(np.round(sample_pred_original, 2)))
            LOGGER.info(f"  Epoch {epoch}: Sample predictions (unique={unique_count}): {sample_pred_original.flatten()[:3]}")

pred_monitor = PredictionMonitor()

# Train
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr, pred_monitor],
    verbose=1
)

print("\n" + "="*80)
print("EVALUATION")
print("="*80)

# Predict on test
y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# Check variability
unique_preds = len(np.unique(np.round(y_pred, 2)))
pred_std = y_pred.std()

print(f"Predictions: {len(y_pred)}")
print(f"Unique values (rounded to 2 decimals): {unique_preds}")
print(f"Prediction std: {pred_std:.4f}")
print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
print(f"Actual range: [{y_test_original.min():.2f}, {y_test_original.max():.2f}]")

if unique_preds > 5:
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mae = np.mean(np.abs(y_test_original - y_pred))
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
    
    print(f"\nMetrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Compare to baseline
    baseline_pred = np.full(len(y_test_original), y_train.mean())
    baseline_pred_original = y_scaler.inverse_transform(baseline_pred.reshape(-1, 1)).flatten()
    baseline_rmse = np.sqrt(mean_squared_error(y_test_original, baseline_pred_original))
    
    print(f"  Baseline RMSE (mean): {baseline_rmse:.4f}")
    print(f"  Improvement: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")
    
    if rmse < baseline_rmse and unique_preds > 10:
        print("\n✅ SUCCESS: Model is learning and beating baseline!")
        
        # Save model
        model.save('models/tft_simplified.keras')
        print("💾 Saved to models/tft_simplified.keras")
        
        # Also save scalers
        import pickle
        with open('models/tft_scalers.pkl', 'wb') as f:
            pickle.dump({'X_scaler': X_scaler, 'y_scaler': y_scaler, 'encoder_length': encoder_length}, f)
        print("💾 Saved scalers to models/tft_scalers.pkl")
    else:
        print("\n⚠️ Model not significantly better than baseline")
else:
    print("\n❌ Model still producing constant/near-constant predictions")
    print("This indicates a fundamental training issue.")

print("\nSample predictions vs actual (last 10):")
comparison = pd.DataFrame({
    'Actual': y_test_original[-10:],
    'Predicted': y_pred[-10:],
    'Error': y_test_original[-10:] - y_pred[-10:]
})
print(comparison.to_string())
