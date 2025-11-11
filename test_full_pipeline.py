"""
Complete End-to-End Test of OER Forecasting Pipeline

This script tests:
1. Data fetching and feature engineering
2. Model training (XGBoost, baselines)
3. Backtesting
4. Interpretability (SHAP)
5. Ensemble methods
6. Performance evaluation
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.oer_model.utils.logging import get_logger
from src.oer_model.config import load_config
from src.oer_model.data.fred import fetch_bulk
from src.oer_model.models.xgboost import XGBoostModel
from src.oer_model.models.baselines import LassoModel
from src.oer_model.evaluation.backtest import run_backtest, BacktestConfig
from src.oer_model.evaluation.reporting import (
    aggregate_backtest_results,
    compute_shap_values,
    plot_shap_summary,
    generate_interpretation_report
)
from src.oer_model.forecasting.ensembles import (
    equal_weight_average,
    performance_weighted_average,
    bayesian_model_averaging,
    stacking_ensemble
)

LOGGER = get_logger(__name__)


def setup_directories():
    """Create necessary output directories."""
    dirs = [
        "data/raw",
        "data/processed",
        "outputs/models",
        "outputs/backtest",
        "outputs/interpretability",
        "outputs/ensembles"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    LOGGER.info("✓ Directories created")


def fetch_and_prepare_data():
    """Load data from existing CSV file."""
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 1: DATA LOADING")
    LOGGER.info("=" * 80)
    
    try:
        # Load the existing CSV file
        csv_path = "oer_forecast_monthly_features.csv"
        LOGGER.info("Loading data from %s...", csv_path)
        
        features_df = pd.read_csv(csv_path)
        
        # Try to identify date column
        date_cols = [col for col in features_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            features_df[date_cols[0]] = pd.to_datetime(features_df[date_cols[0]])
            features_df = features_df.set_index(date_cols[0])
        elif features_df.columns[0] not in ['Unnamed: 0']:
            # Try first column as date
            try:
                features_df[features_df.columns[0]] = pd.to_datetime(features_df[features_df.columns[0]])
                features_df = features_df.set_index(features_df.columns[0])
            except:
                pass
        
        # Drop unnamed columns
        features_df = features_df.loc[:, ~features_df.columns.str.contains('^Unnamed')]
        
        LOGGER.info("✓ Loaded data: %d rows, %d columns", len(features_df), len(features_df.columns))
        LOGGER.info("✓ Columns: %s", list(features_df.columns[:10]))
        
        if len(features_df) == 0:
            LOGGER.error("Loaded dataframe is empty!")
            return None
        
        # Save a copy to processed
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        features_df.to_csv("data/processed/features.csv")
        LOGGER.info("✓ Saved copy to data/processed/features.csv")
        
        return features_df
        
    except Exception as e:
        LOGGER.error("Data loading failed: %s", e)
        import traceback
        traceback.print_exc()
        return None


def train_models(features_df):
    """Train multiple models."""
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 2: MODEL TRAINING")
    LOGGER.info("=" * 80)
    
    # Prepare data - use CPIQOEPS Index as target
    target_col = 'CPIQOEPS Index'
    
    if target_col not in features_df.columns:
        LOGGER.error("Target column '%s' not found in columns: %s", target_col, list(features_df.columns[:10]))
        return {}
    
    LOGGER.info("Using target column: %s", target_col)
    
    # Drop rows with missing target
    data = features_df.dropna(subset=[target_col])
    
    # Split features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Drop columns with too many missing values
    missing_pct = X.isnull().sum() / len(X)
    X = X.loc[:, missing_pct < 0.3]
    
    # Fill remaining NaNs with forward fill then 0
    X = X.fillna(method='ffill').fillna(0)
    
    LOGGER.info("Training data: %d samples, %d features", len(X), len(X.columns))
    LOGGER.info("Target range: [%.2f, %.2f], mean: %.2f", 
               y.min(), y.max(), y.mean())
    
    # Train-test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    LOGGER.info("Train: %d samples | Test: %d samples", len(X_train), len(X_test))
    
    models = {}
    
    # 1. LASSO Baseline
    LOGGER.info("\n--- Training LASSO Baseline ---")
    try:
        lasso = LassoModel(name="lasso")
        lasso.fit(X_train, y_train)
        
        train_pred = lasso.predict(X_train)
        test_pred = lasso.predict(X_test)
        
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
        
        LOGGER.info("✓ LASSO trained - Train RMSE: %.4f, Test RMSE: %.4f", 
                   train_rmse, test_rmse)
        models['lasso'] = lasso
    except Exception as e:
        LOGGER.error("LASSO training failed: %s", e)
    
    # 2. XGBoost
    LOGGER.info("\n--- Training XGBoost ---")
    try:
        xgb = XGBoostModel(
            name="xgboost",
            params={
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        )
        xgb.fit(X_train, y_train)
        
        train_pred = xgb.predict(X_train)
        test_pred = xgb.predict(X_test)
        
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
        
        LOGGER.info("✓ XGBoost trained - Train RMSE: %.4f, Test RMSE: %.4f", 
                   train_rmse, test_rmse)
        models['xgboost'] = xgb
    except Exception as e:
        LOGGER.error("XGBoost training failed: %s", e)
    
    # Store data for later use
    models['_data'] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X': X,
        'y': y
    }
    
    LOGGER.info("\n✓ Model training complete: %d models trained", len(models) - 1)
    return models


def run_backtests(models):
    """Run rolling window backtests."""
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 3: BACKTESTING")
    LOGGER.info("=" * 80)
    
    data = models['_data']
    X, y = data['X'], data['y']
    
    backtest_results = {}
    
    for model_name, model in models.items():
        if model_name == '_data':
            continue
            
        LOGGER.info("\n--- Backtesting %s ---", model_name.upper())
        
        try:
            # Prepare data with target
            data_with_target = X.copy()
            data_with_target['target'] = y
            
            # Configure backtest
            cfg = BacktestConfig(
                train_window=120,  # 10 years
                test_window=12,    # 1 year test
                step=30,           # Move roughly one month at a time
                max_windows=60,    # Limit number of windows during debugging
                log_frequency=5    # Log every 5th window for brevity
            )
            
            # Create model factory based on model type
            if model_name == 'lasso':
                model_factory = lambda: LassoModel(name=model_name)
            elif model_name == 'xgboost':
                from src.oer_model.models.xgboost import XGBParams
                xgb_params = XGBParams(
                    objective='reg:squarederror',
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4
                )
                model_factory = lambda: XGBoostModel(name=model_name, params=xgb_params)
            else:
                # Generic factory
                model_factory = lambda: model.__class__(name=model.name)
            
            # Run backtest
            results = run_backtest(
                data=data_with_target,
                target_column='target',
                model_factory=model_factory,
                cfg=cfg
            )
            
            # Calculate metrics
            results['error'] = results['y_true'] - results['y_pred']
            results['abs_error'] = results['error'].abs()
            results['squared_error'] = results['error'] ** 2
            
            rmse = np.sqrt(results['squared_error'].mean())
            mae = results['abs_error'].mean()
            mape = (results['abs_error'] / results['y_true'].abs()).mean() * 100
            
            LOGGER.info("  RMSE: %.4f", rmse)
            LOGGER.info("  MAE:  %.4f", mae)
            LOGGER.info("  MAPE: %.2f%%", mape)
            
            backtest_results[model_name] = results
            
            # Save results
            results.to_csv(f"outputs/backtest/{model_name}_backtest.csv", index=False)
            
        except Exception as e:
            LOGGER.error("Backtest failed for %s: %s", model_name, e)
    
    # Aggregate results
    if backtest_results:
        LOGGER.info("\n--- Performance Summary ---")
        all_results = []
        for model_name, results in backtest_results.items():
            results['model'] = model_name
            all_results.append(results)
        
        combined = pd.concat(all_results, ignore_index=True)
        combined['rmse'] = np.sqrt(combined['squared_error'])
        combined['mae'] = combined['abs_error']
        combined['mape'] = (combined['abs_error'] / combined['y_true'].abs()) * 100
        
        summary = aggregate_backtest_results(combined)
        LOGGER.info("\n%s", summary.to_string())
        
        summary.to_csv("outputs/backtest/summary.csv")
    
    return backtest_results


def generate_interpretability_reports(models):
    """Generate SHAP and other interpretability reports."""
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 4: INTERPRETABILITY ANALYSIS")
    LOGGER.info("=" * 80)
    
    data = models['_data']
    X_test, y_test = data['X_test'], data['y_test']
    
    for model_name, model in models.items():
        if model_name == '_data':
            continue
        
        LOGGER.info("\n--- Generating interpretation for %s ---", model_name.upper())
        
        try:
            output_dir = Path(f"outputs/interpretability/{model_name}")
            
            # Use underlying estimator when available (e.g., scikit-learn, XGBoost)
            base_model = getattr(model, 'estimator', model)
            model_class_name = base_model.__class__.__name__.lower()
            
            if any(keyword in model_class_name for keyword in ('xgb', 'boost', 'tree')):
                model_type = "tree"
            elif any(keyword in model_class_name for keyword in ('lasso', 'linear', 'ridge', 'elastic')):
                model_type = "linear"
            else:
                model_type = "auto"
            
            artifacts = generate_interpretation_report(
                model=base_model,
                X=X_test,
                y=y_test,
                model_name=model_name,
                output_dir=output_dir,
                model_type=model_type
            )
            
            LOGGER.info("✓ Generated %d interpretation artifacts", len(artifacts))
            for artifact_name, artifact_path in artifacts.items():
                LOGGER.info("  - %s", artifact_path.name)
                
        except Exception as e:
            LOGGER.error("Interpretation failed for %s: %s", model_name, e)


def create_ensembles(models, backtest_results):
    """Create and evaluate ensemble forecasts."""
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("STEP 5: ENSEMBLE FORECASTING")
    LOGGER.info("=" * 80)
    
    if len(backtest_results) < 2:
        LOGGER.warning("Need at least 2 models for ensemble. Skipping.")
        return
    
    data = models['_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Collect forecasts and errors
    test_forecasts = pd.DataFrame(index=X_test.index)
    test_errors = pd.DataFrame(index=X_test.index)
    
    for model_name, model in models.items():
        if model_name == '_data':
            continue
        test_forecasts[model_name] = model.predict(X_test)
        test_errors[model_name] = y_test.values - test_forecasts[model_name].values
    
    LOGGER.info("Forecasts collected from %d models", len(test_forecasts.columns))
    
    # For stacking, we need train forecasts too
    train_forecasts = pd.DataFrame(index=X_train.index)
    for model_name, model in models.items():
        if model_name == '_data':
            continue
        train_forecasts[model_name] = model.predict(X_train)
    
    ensemble_results = {}
    
    # 1. Equal Weight
    LOGGER.info("\n--- Equal Weight Ensemble ---")
    try:
        equal_ens = equal_weight_average(test_forecasts)
        rmse = np.sqrt(np.mean((y_test - equal_ens) ** 2))
        mae = np.mean(np.abs(y_test - equal_ens))
        LOGGER.info("  RMSE: %.4f, MAE: %.4f", rmse, mae)
        ensemble_results['equal_weight'] = equal_ens
    except Exception as e:
        LOGGER.error("Equal weight failed: %s", e)
    
    # 2. Performance Weighted
    LOGGER.info("\n--- Performance Weighted Ensemble (Inverse RMSE) ---")
    try:
        perf_ens = performance_weighted_average(
            test_forecasts,
            test_errors,
            method="inverse_rmse"
        )
        rmse = np.sqrt(np.mean((y_test - perf_ens) ** 2))
        mae = np.mean(np.abs(y_test - perf_ens))
        LOGGER.info("  RMSE: %.4f, MAE: %.4f", rmse, mae)
        ensemble_results['performance_weighted'] = perf_ens
    except Exception as e:
        LOGGER.error("Performance weighted failed: %s", e)
    
    # 3. Bayesian Model Averaging
    LOGGER.info("\n--- Bayesian Model Averaging ---")
    try:
        bma_ens = bayesian_model_averaging(
            test_forecasts,
            test_errors,
            prior=None,
            temperature=1.0
        )
        rmse = np.sqrt(np.mean((y_test - bma_ens) ** 2))
        mae = np.mean(np.abs(y_test - bma_ens))
        LOGGER.info("  RMSE: %.4f, MAE: %.4f", rmse, mae)
        ensemble_results['bma'] = bma_ens
    except Exception as e:
        LOGGER.error("BMA failed: %s", e)
    
    # 4. Stacking (Ridge)
    LOGGER.info("\n--- Stacking Ensemble (Ridge) ---")
    try:
        stacked_ens = stacking_ensemble(
            train_forecasts,
            y_train,
            test_forecasts,
            meta_model="ridge",
            alpha=1.0
        )
        rmse = np.sqrt(np.mean((y_test - stacked_ens) ** 2))
        mae = np.mean(np.abs(y_test - stacked_ens))
        LOGGER.info("  RMSE: %.4f, MAE: %.4f", rmse, mae)
        ensemble_results['stacking'] = stacked_ens
    except Exception as e:
        LOGGER.error("Stacking failed: %s", e)
    
    # Save ensemble results
    ensemble_df = pd.DataFrame(ensemble_results, index=X_test.index)
    ensemble_df['actual'] = y_test.values
    ensemble_df.to_csv("outputs/ensembles/ensemble_forecasts.csv")
    
    # Save individual model forecasts for diagnostics
    model_forecasts = test_forecasts.copy()
    model_forecasts['actual'] = y_test.values
    model_forecasts.to_csv("outputs/ensembles/model_test_forecasts.csv")
    
    # Summary comparison
    LOGGER.info("\n--- Ensemble Performance Summary ---")
    summary_data = []
    for ens_name, ens_forecast in ensemble_results.items():
        rmse = np.sqrt(np.mean((y_test - ens_forecast) ** 2))
        mae = np.mean(np.abs(y_test - ens_forecast))
        mape = np.mean(np.abs((y_test - ens_forecast) / y_test)) * 100
        summary_data.append({
            'Method': ens_name,
            'RMSE': f"{rmse:.4f}",
            'MAE': f"{mae:.4f}",
            'MAPE': f"{mape:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    LOGGER.info("\n%s", summary_df.to_string(index=False))
    
    summary_df.to_csv("outputs/ensembles/ensemble_summary.csv", index=False)
    
    return ensemble_results, model_forecasts, summary_df


def main():
    """Run complete pipeline test."""
    start_time = datetime.now()
    
    LOGGER.info("=" * 80)
    LOGGER.info("OER FORECASTING PIPELINE - COMPLETE TEST")
    LOGGER.info("=" * 80)
    LOGGER.info("Start time: %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("")
    
    # Step 0: Setup
    setup_directories()
    
    # Step 1: Data
    features_df = fetch_and_prepare_data()
    if features_df is None:
        LOGGER.error("Pipeline failed at data step")
        return
    
    # Step 2: Train models
    models = train_models(features_df)
    if not models or len(models) <= 1:  # Only _data key
        LOGGER.error("Pipeline failed at training step")
        return
    
    # Step 3: Backtest
    backtest_results = run_backtests(models)
    
    # Step 4: Interpretability
    generate_interpretability_reports(models)
    
    # Step 5: Ensembles
    if backtest_results:
        create_ensembles(models, backtest_results)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("PIPELINE TEST COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("End time: %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("Duration: %s", duration)
    LOGGER.info("")
    LOGGER.info("Results saved to:")
    LOGGER.info("  - data/processed/features.csv")
    LOGGER.info("  - outputs/backtest/")
    LOGGER.info("  - outputs/interpretability/")
    LOGGER.info("  - outputs/ensembles/")
    LOGGER.info("")


if __name__ == "__main__":
    main()
