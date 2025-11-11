"""
Diagnose and Fix TFT Constant Prediction Issue

This script:
1. Tests different encoder lengths (6, 12, 18)
2. Evaluates each on validation set
3. Checks for actual learning vs mean prediction
4. Recommends optimal configuration
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from oer_model.models.tft_tensorflow import TFTTensorFlowModel
from oer_model.utils.io import read_dataframe
import pickle

print("="*80)
print("TFT DIAGNOSTIC TEST - FINDING OPTIMAL ENCODER LENGTH")
print("="*80)

# Load data
df = read_dataframe('data/processed/features_processed.csv').set_index('date')
X = df[[c for c in df.columns if c != 'oer_cpi_yoy']]
y = df['oer_cpi_yoy']

print(f"\nDataset: {len(X)} samples, {len(X.columns)} features")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# Baseline: predict mean
mean_prediction = np.full(len(y_test), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_prediction))
print(f"\nBaseline (predict mean): RMSE = {baseline_rmse:.4f}")

print("\n" + "="*80)
print("TESTING DIFFERENT ENCODER LENGTHS")
print("="*80)

results = []

for encoder_length in [6, 12, 18]:
    print(f"\n--- Testing encoder_length = {encoder_length} ---")
    
    # Check if we have enough data
    min_samples = encoder_length + 10
    if len(X_train) < min_samples:
        print(f"  ⚠️ SKIP: Need at least {min_samples} samples, have {len(X_train)}")
        continue
    
    try:
        # Train model with longer epochs and different learning rate
        params = {
            'max_encoder_length': encoder_length,
            'hidden_size': 64,
            'attention_head_size': 4,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,  # Smaller batch for better gradients
            'max_epochs': 100  # More epochs
        }
        
        model = TFTTensorFlowModel(name='tft', params=params)
        print(f"  Training with {params}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Filter out NaN predictions
        valid_mask = ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            print(f"  ❌ FAIL: All predictions are NaN")
            continue
        
        y_test_valid = y_test.values[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # Check prediction variability
        unique_preds = len(np.unique(np.round(y_pred_valid, 4)))
        pred_std = y_pred_valid.std()
        
        print(f"  Valid predictions: {len(y_pred_valid)}/{len(y_test)}")
        print(f"  Unique prediction values: {unique_preds}")
        print(f"  Prediction std: {pred_std:.4f}")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))
        mae = mean_absolute_error(y_test_valid, y_pred_valid)
        mape = np.mean(np.abs((y_test_valid - y_pred_valid) / y_test_valid)) * 100
        
        # Check if model learned anything
        if unique_preds <= 1:
            status = "❌ CONSTANT (no learning)"
        elif rmse >= baseline_rmse:
            status = "⚠️ WORSE THAN MEAN"
        elif pred_std < 0.1:
            status = "⚠️ LOW VARIABILITY"
        else:
            status = "✅ LEARNING"
        
        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")
        print(f"  Status: {status}")
        
        results.append({
            'encoder_length': encoder_length,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'unique_preds': unique_preds,
            'pred_std': pred_std,
            'valid_preds': len(y_pred_valid),
            'status': status,
            'better_than_baseline': rmse < baseline_rmse
        })
        
        # Save if this is best so far
        if results and results[-1]['rmse'] == min(r['rmse'] for r in results):
            print(f"  💾 Saving as best model so far...")
            with open('models/tft_best.pkl', 'wb') as f:
                pickle.dump(model, f)
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Find best
    best = results_df.loc[results_df['rmse'].idxmin()]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   encoder_length: {int(best['encoder_length'])}")
    print(f"   RMSE: {best['rmse']:.4f}")
    print(f"   Unique predictions: {int(best['unique_preds'])}")
    print(f"   Better than baseline: {best['better_than_baseline']}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if best['rmse'] < baseline_rmse:
        print(f"   ✅ Use encoder_length={int(best['encoder_length'])} in config.yaml")
        print(f"   ✅ Model is learning and beats mean baseline")
    else:
        print(f"   ⚠️ TFT not beating baseline - needs hyperparameter tuning")
        print(f"   ⚠️ Consider: lower learning_rate (0.0001), larger hidden_size (96-128)")
        print(f"   ⚠️ Or use XGBoost (already working well)")
else:
    print("❌ No successful configurations found")
    print("\nTroubleshooting:")
    print("  1. Dataset too small (need 100+ samples)")
    print("  2. Features not normalized properly")
    print("  3. TFT architecture needs adjustment")
    print("  4. Consider using simpler models (XGBoost, LASSO)")

print("\n" + "="*80)
