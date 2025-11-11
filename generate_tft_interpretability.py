"""Generate TFT interpretability artifacts: attention weights and variable importance."""
import sys
sys.path.insert(0, 'src')

import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Load TFT model
model_path = 'models/tft.pkl'
if not Path(model_path).exists():
    LOGGER.error("TFT model not found. Train the model first.")
    sys.exit(1)

with open(model_path, 'rb') as f:
    tft_model = pickle.load(f)

LOGGER.info("TFT model loaded successfully")

# Load data
from oer_model.utils.io import read_dataframe
df = read_dataframe('data/processed/features_processed.csv')
df = df.set_index('date')

feature_cols = [c for c in df.columns if c != 'oer_cpi_yoy']
X = df[feature_cols].values

# Get encoder length
encoder_length = tft_model.params.get('max_encoder_length', 12)

# Create a single sequence for interpretation
seq_idx = len(X) - encoder_length
X_seq = X[seq_idx:seq_idx + encoder_length].reshape(1, encoder_length, -1)

# Scale inputs
X_seq_scaled = tft_model._input_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1]))
X_seq_scaled = X_seq_scaled.reshape(1, encoder_length, -1)

LOGGER.info(f"Input sequence shape: {X_seq_scaled.shape}")

# Extract variable selection weights
# The TFT model's variable_selection layer computes feature importance
import tensorflow as tf

# Get the TFT layer
tft_layer = tft_model._model.layers[1]  # TemporalFusionTransformerTF layer

# Run forward pass and extract variable weights
output, var_weights = tft_layer(tf.constant(X_seq_scaled, dtype=tf.float32), training=False)

var_weights_np = var_weights.numpy()[0, 0, :]  # (features,)

LOGGER.info(f"Variable weights shape: {var_weights_np.shape}")

# Create variable importance dataframe
var_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': var_weights_np
}).sort_values('importance', ascending=False)

# Save to artifacts
artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(exist_ok=True)

var_importance.to_csv(artifacts_dir / 'tft_variable_importance.csv', index=False)
LOGGER.info("Saved TFT variable importance to artifacts/tft_variable_importance.csv")

# Create visualization
plt.figure(figsize=(12, 8))
top_n = 20
top_features = var_importance.head(top_n)

sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title(f'TFT Top {top_n} Variable Importance (Variable Selection Weights)', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(artifacts_dir / 'tft_variable_importance.png', dpi=300, bbox_inches='tight')
LOGGER.info("Saved TFT variable importance plot to artifacts/tft_variable_importance.png")

print("\n" + "="*80)
print("TFT VARIABLE IMPORTANCE (Top 15)")
print("="*80)
print(var_importance.head(15).to_string(index=False))
print("\n")

# Try to extract attention weights if available
try:
    # Get attention layer output
    attention_layer = None
    for layer in tft_model._model.layers:
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if 'attention' in sublayer.name.lower():
                    attention_layer = sublayer
                    break
    
    if attention_layer:
        LOGGER.info("Found attention layer, extracting weights...")
        # This would require modifying the model to output attention weights
        # For now, we'll note that this requires model architecture changes
        LOGGER.warning("Attention weight extraction requires model modifications (output attention_scores)")
    else:
        LOGGER.info("No separate attention layer found (attention integrated in TFT layer)")
except Exception as e:
    LOGGER.warning(f"Could not extract attention weights: {e}")

print("="*80)
print("TFT interpretability artifacts generated successfully!")
print(f"- Variable importance: artifacts/tft_variable_importance.csv")
print(f"- Importance plot: artifacts/tft_variable_importance.png")
print("="*80)
