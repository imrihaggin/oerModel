"""Tune TFT hyperparameters using Optuna."""
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from oer_model.models.tft_tensorflow import TFTTensorFlowModel
from oer_model.utils.io import read_dataframe, save_json

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features_processed.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

if not FEATURES_PATH.exists():
    raise SystemExit("Features not found. Run build-features first.")

# Load data
df = read_dataframe(str(FEATURES_PATH))
df = df.set_index('date')

feature_cols = [c for c in df.columns if c != 'oer_cpi_yoy']
X = df[feature_cols]
y = df['oer_cpi_yoy']

LOGGER.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Objective function
def objective(trial):
    """Optuna objective for TFT hyperparameters."""
    
    # Hyperparameter search space
    params = {
        'max_encoder_length': trial.suggest_categorical('max_encoder_length', [6, 12, 18, 24]),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 48, 64, 96]),
        'attention_head_size': trial.suggest_categorical('attention_head_size', [2, 4, 6]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'max_epochs': 50  # Fixed, early stopping will handle it
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Need enough samples for encoder length
            if len(X_train) < params['max_encoder_length'] + 10:
                LOGGER.warning(f"Fold {fold_idx}: insufficient training samples, skipping")
                continue
            
            # Train model
            model = TFTTensorFlowModel(name='tft', params=params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_scores.append(rmse)
            
            LOGGER.info(f"Trial {trial.number} Fold {fold_idx}: RMSE={rmse:.4f}")
            
        except Exception as e:
            LOGGER.error(f"Trial {trial.number} Fold {fold_idx} failed: {e}")
            continue
    
    if not cv_scores:
        return float('inf')
    
    mean_rmse = np.mean(cv_scores)
    LOGGER.info(f"Trial {trial.number} Mean CV RMSE: {mean_rmse:.4f}")
    
    return mean_rmse

# Create study
study = optuna.create_study(
    direction='minimize',
    study_name='tft_tuning',
    sampler=optuna.samplers.TPESampler(seed=42)
)

LOGGER.info("Starting Optuna hyperparameter search...")
LOGGER.info("Search space:")
LOGGER.info("  max_encoder_length: [6, 12, 18, 24]")
LOGGER.info("  hidden_size: [32, 48, 64, 96]")
LOGGER.info("  attention_head_size: [2, 4, 6]")
LOGGER.info("  dropout: [0.1, 0.3]")
LOGGER.info("  learning_rate: [1e-4, 1e-2] (log scale)")
LOGGER.info("  batch_size: [16, 32, 64]")
LOGGER.info("")

# Run optimization (adjust n_trials as needed)
n_trials = 30
study.optimize(objective, n_trials=n_trials, n_jobs=1)

# Best parameters
best_params = study.best_params
best_score = study.best_value

LOGGER.info("\n" + "="*80)
LOGGER.info("BEST HYPERPARAMETERS")
LOGGER.info("="*80)
for param, value in best_params.items():
    LOGGER.info(f"  {param}: {value}")
LOGGER.info(f"\nBest CV RMSE: {best_score:.4f}")
LOGGER.info("="*80)

# Save results
result = {
    'best_params': best_params,
    'best_cv_rmse': float(best_score),
    'n_trials': n_trials,
    'study_name': 'tft_tuning'
}

save_json(result, ARTIFACTS / 'tft_optuna_params.json')
LOGGER.info(f"\nSaved best parameters to artifacts/tft_optuna_params.json")

# Save trials dataframe
trials_df = study.trials_dataframe()
trials_df.to_csv(ARTIFACTS / 'tft_optuna_trials.csv', index=False)
LOGGER.info(f"Saved all trials to artifacts/tft_optuna_trials.csv")

print("\n" + "="*80)
print("TFT HYPERPARAMETER TUNING COMPLETE")
print("="*80)
print(f"Best CV RMSE: {best_score:.4f}")
print(f"\nUpdate config/config.yaml with these parameters:")
print("models:")
print("  candidates:")
print("    tft:")
print("      params:")
for param, value in best_params.items():
    if param == 'learning_rate':
        print(f"        {param}: {value:.6f}")
    else:
        print(f"        {param}: {value}")
print("="*80)
