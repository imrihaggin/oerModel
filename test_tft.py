"""Quick test of TFT TensorFlow model."""
import sys
sys.path.insert(0, 'src')

import logging
import pandas as pd
from oer_model.models.tft_tensorflow import TFTTensorFlowModel
from oer_model.utils.io import read_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
df = read_dataframe('data/processed/features_processed.csv')
df = df.set_index('date')

# Select features and target
feature_cols = [c for c in df.columns if c != 'oer_cpi_yoy']
X = df[feature_cols]
y = df['oer_cpi_yoy']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Train TFT
params = {
    'max_encoder_length': 12,  # Reduced for faster training
    'hidden_size': 32,  # Reduced
    'attention_head_size': 2,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 16,
    'max_epochs': 20  # Reduced
}

model = TFTTensorFlowModel(name='tft', params=params)
print("\nStarting TFT training...")
model.fit(X, y)

print("\nGenerating predictions...")
preds = model.predict(X)
print(f"Predictions shape: {preds.shape}")
print(f"Predictions (last 10): {preds[-10:]}")

# Save model
import pickle
with open('models/tft.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to models/tft.pkl")
