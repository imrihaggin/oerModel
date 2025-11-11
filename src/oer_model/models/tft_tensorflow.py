"""TensorFlow-based Temporal Fusion Transformer implementation."""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .base import ForecastModel

LOGGER = logging.getLogger(__name__)


class VariableSelectionNetwork(layers.Layer):
    """Variable selection network for TFT."""
    
    def __init__(self, num_features, hidden_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # GRN for each variable
        self.grn_var = layers.Dense(hidden_size, activation='elu')
        self.grn_dropout = layers.Dropout(dropout_rate)
        
        # Variable selection weights
        self.variable_weights = layers.Dense(num_features, activation='softmax')
        
    def call(self, inputs, training=None):
        # inputs: (batch, time, features)
        # Transform each variable
        transformed = self.grn_var(inputs)
        transformed = self.grn_dropout(transformed, training=training)
        
        # Compute variable selection weights
        # Average over time dimension to get feature importance
        flattened = tf.reduce_mean(transformed, axis=1)
        weights = self.variable_weights(flattened)
        weights = tf.expand_dims(weights, axis=1)  # (batch, 1, features)
        
        # Apply weights to inputs
        selected = inputs * weights
        return selected, weights


class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network (GRN) component."""
    
    def __init__(self, hidden_size, output_size=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(hidden_size, activation='elu')
        self.dense2 = layers.Dense(self.output_size)
        self.dropout = layers.Dropout(dropout_rate)
        self.gate = layers.Dense(self.output_size, activation='sigmoid')
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        # Primary path
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        # Gating
        gate = self.gate(inputs)
        gated = x * gate
        
        # Residual connection (if dimensions match)
        if inputs.shape[-1] == self.output_size:
            output = self.layer_norm(inputs + gated)
        else:
            output = self.layer_norm(gated)
            
        return output


class TemporalFusionTransformerTF(layers.Layer):
    """TensorFlow implementation of Temporal Fusion Transformer."""
    
    def __init__(self, 
                 num_features: int,
                 encoder_length: int,
                 hidden_size: int = 64,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.encoder_length = encoder_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        
        # Variable selection
        self.variable_selection = VariableSelectionNetwork(
            num_features, hidden_size, dropout_rate
        )
        
        # LSTM encoders
        self.encoder_lstm = layers.LSTM(
            hidden_size, 
            return_sequences=True, 
            return_state=True,
            dropout=dropout_rate
        )
        
        # Gated residual networks
        self.grn_encoder = GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate)
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_size // num_attention_heads,
            dropout=dropout_rate
        )
        
        # Post-attention processing
        self.attention_dropout = layers.Dropout(dropout_rate)
        self.attention_norm = layers.LayerNormalization()
        
        # Position-wise feed forward
        self.ff_grn = GatedResidualNetwork(hidden_size, dropout_rate=dropout_rate)
        
        # Output
        self.output_grn = GatedResidualNetwork(hidden_size, output_size=1, dropout_rate=dropout_rate)
        
    def call(self, inputs, training=None):
        # inputs: (batch, encoder_length, num_features)
        
        # Variable selection
        selected, var_weights = self.variable_selection(inputs, training=training)
        
        # LSTM encoding
        lstm_out, *lstm_states = self.encoder_lstm(selected, training=training)
        
        # Apply GRN to LSTM output
        encoded = self.grn_encoder(lstm_out, training=training)
        
        # Multi-head self-attention
        attention_out = self.attention(
            encoded, encoded, encoded,
            training=training
        )
        attention_out = self.attention_dropout(attention_out, training=training)
        attention_out = self.attention_norm(encoded + attention_out)
        
        # Position-wise feed forward
        ff_out = self.ff_grn(attention_out, training=training)
        
        # Take last timestep and produce output
        last_step = ff_out[:, -1, :]
        output = self.output_grn(last_step, training=training)
        
        return output, var_weights


class TFTTensorFlowModel(ForecastModel):
    """Temporal Fusion Transformer using TensorFlow/Keras."""
    
    def __init__(self, name: str = 'tft', params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        self.params = params or {}
        self._model: Optional[keras.Model] = None
        self._input_scaler = None
        self._output_scaler = None
        self._feature_columns = None
        self._trained = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the TFT model."""
        LOGGER.info("Training TFT (TensorFlow) with %d samples", len(X))
        
        # Store feature columns
        self._feature_columns = X.columns.tolist()
        
        # Normalize inputs and outputs
        from sklearn.preprocessing import StandardScaler
        self._input_scaler = StandardScaler()
        self._output_scaler = StandardScaler()
        
        X_scaled = self._input_scaler.fit_transform(X.values)
        y_scaled = self._output_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Extract parameters
        encoder_length = self.params.get('max_encoder_length', 12)
        hidden_size = self.params.get('hidden_size', 64)
        num_attention_heads = self.params.get('attention_head_size', 4)
        dropout_rate = self.params.get('dropout', 0.1)
        batch_size = self.params.get('batch_size', 32)
        max_epochs = self.params.get('max_epochs', 50)
        learning_rate = self.params.get('learning_rate', 0.001)
        
        # Prepare sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled, encoder_length)
        
        LOGGER.info(f"Created {len(X_seq)} sequences of length {encoder_length}")
        
        # Build model
        num_features = X.shape[1]
        
        input_layer = keras.Input(shape=(encoder_length, num_features), name='encoder_input')
        tft_output, var_weights = TemporalFusionTransformerTF(
            num_features=num_features,
            encoder_length=encoder_length,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate
        )(input_layer, training=True)
        
        self._model = keras.Model(inputs=input_layer, outputs=tft_output, name='tft_model')
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        LOGGER.info("Model summary:")
        self._model.summary(print_fn=LOGGER.info)
        
        # Train
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        )
        
        # Split into train/val
        val_split = 0.2
        val_size = int(len(X_seq) * val_split)
        X_train, X_val = X_seq[:-val_size], X_seq[-val_size:]
        y_train, y_val = y_seq[:-val_size], y_seq[-val_size:]
        
        LOGGER.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        history = self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self._trained = True
        LOGGER.info(f"Training completed. Final train loss: {history.history['loss'][-1]:.4f}, "
                   f"val loss: {history.history['val_loss'][-1]:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")
            
        encoder_length = self.params.get('max_encoder_length', 12)
        
        # Check if we have enough data
        if len(X) < encoder_length:
            LOGGER.warning(f"Insufficient data for prediction: {len(X)} samples < {encoder_length} required. Returning NaNs.")
            return np.full(len(X), np.nan)
        
        # Scale inputs
        X_scaled = self._input_scaler.transform(X.values)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)), encoder_length)
        
        # Check if sequences were created
        if len(X_seq) == 0:
            LOGGER.warning(f"No sequences created for prediction. Returning NaNs.")
            return np.full(len(X), np.nan)
        
        # Predict
        y_pred_scaled = self._model.predict(X_seq, verbose=0)
        
        # Inverse transform
        y_pred = self._output_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Pad to match input length
        padding = np.full(encoder_length - 1, np.nan)
        y_pred_full = np.concatenate([padding, y_pred])
        
        return y_pred_full
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        """Create sequences for time series prediction."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length - 1])
            
        return np.array(X_seq), np.array(y_seq)
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self.params
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        self.params.update(params)
    
    def __getstate__(self):
        """Custom pickle serialization."""
        state = self.__dict__.copy()
        # Save Keras model to temp bytes instead of pickling
        if self._model is not None:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                tmp_path = tmp.name
            try:
                self._model.save(tmp_path)
                with open(tmp_path, 'rb') as f:
                    model_bytes = f.read()
                state['_model_bytes'] = model_bytes
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            state['_model'] = None  # Don't pickle the model object
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization."""
        # Restore model from bytes
        if '_model_bytes' in state:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                tmp.write(state['_model_bytes'])
                tmp_path = tmp.name
            try:
                state['_model'] = keras.models.load_model(tmp_path, custom_objects={
                    'VariableSelectionNetwork': VariableSelectionNetwork,
                    'GatedResidualNetwork': GatedResidualNetwork,
                    'TemporalFusionTransformerTF': TemporalFusionTransformerTF
                })
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            del state['_model_bytes']
        self.__dict__.update(state)
