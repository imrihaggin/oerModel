"""
Example: Using Interpretability and Advanced Ensembles

This example demonstrates:
1. SHAP value computation for tree-based models (XGBoost)
2. TFT attention heatmap generation
3. Bayesian Model Averaging for ensemble forecasting
4. Performance-weighted ensemble forecasting
5. Creating comprehensive interpretation reports
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.oer_model.models.xgboost import XGBoostModel
from src.oer_model.models.tft_interpretable import InterpretableTFT
from src.oer_model.evaluation.reporting import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
    extract_tft_attention,
    plot_tft_attention_heatmap,
    generate_interpretation_report,
    create_model_comparison_report
)
from src.oer_model.forecasting.ensembles import (
    equal_weight_average,
    performance_weighted_average,
    bayesian_model_averaging,
    stacking_ensemble,
    dynamic_weighted_ensemble,
    optimal_ensemble
)
from src.oer_model.utils.logging import get_logger

LOGGER = get_logger(__name__)


def example_shap_interpretation():
    """Demonstrate SHAP interpretation for XGBoost model."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 1: SHAP Interpretation for Tree-Based Models")
    LOGGER.info("=" * 80)
    
    # Load some sample data
    data = pd.read_csv("data/processed/features.csv", parse_dates=['date'])
    data = data.set_index('date')
    
    # Split into features and target
    target_col = 'oer_yoy'
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split into train/test
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train XGBoost model
    LOGGER.info("Training XGBoost model...")
    model = XGBoostModel(
        name="xgboost_shap_example",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05
    )
    model.fit(X_train, y_train)
    
    # Compute SHAP values
    LOGGER.info("Computing SHAP values...")
    shap_result = compute_shap_values(
        model=model.model_,  # Access underlying XGBoost model
        X=X_test,
        model_type="tree",
        max_samples=500
    )
    
    if shap_result is None:
        LOGGER.error("SHAP computation failed. Install SHAP: pip install shap")
        return
    
    # Create output directory
    output_dir = Path("outputs/interpretability/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SHAP plots
    LOGGER.info("Generating SHAP visualizations...")
    
    # Summary plot (beeswarm)
    plot_shap_summary(
        shap_result,
        output_path=output_dir / "shap_summary.png",
        plot_type="dot"
    )
    
    # Bar plot (feature importance)
    plot_shap_summary(
        shap_result,
        output_path=output_dir / "shap_importance.png",
        plot_type="bar"
    )
    
    # Waterfall plot for most recent prediction
    plot_shap_waterfall(
        shap_result,
        sample_idx=-1,
        output_path=output_dir / "shap_waterfall_latest.png"
    )
    
    LOGGER.info("SHAP interpretation complete. Outputs saved to %s", output_dir)
    LOGGER.info("")


def example_tft_attention():
    """Demonstrate TFT attention extraction and visualization."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 2: TFT Attention Heatmap")
    LOGGER.info("=" * 80)
    
    # Load data
    data = pd.read_csv("data/processed/features.csv", parse_dates=['date'])
    data = data.set_index('date')
    
    # Prepare for TFT (requires specific format)
    target_col = 'oer_yoy'
    time_varying_known = ['month', 'quarter']  # Known future values
    time_varying_unknown = [col for col in data.columns 
                           if col != target_col and col not in time_varying_known]
    
    # Train TFT model
    LOGGER.info("Training TFT model...")
    tft_model = InterpretableTFT(
        name="tft_attention_example",
        max_epochs=50,
        batch_size=64,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        learning_rate=0.001
    )
    
    # Note: TFT requires PyTorch Forecasting data format
    # This is simplified for demonstration
    # In practice, use TimeSeriesDataSet from pytorch_forecasting
    
    # Extract attention (assuming model is fitted)
    LOGGER.info("Extracting attention patterns...")
    attention_result = extract_tft_attention(
        tft_model=tft_model,
        X=data,
        sample_idx=-1  # Latest prediction
    )
    
    if attention_result is None:
        LOGGER.warning("Attention extraction failed. Model may need fitting.")
        return
    
    # Create output directory
    output_dir = Path("outputs/interpretability/tft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate attention heatmap
    LOGGER.info("Generating attention heatmap...")
    plot_tft_attention_heatmap(
        attention_result,
        output_path=output_dir / "attention_heatmap.png",
        figsize=(14, 8),
        cmap='YlOrRd'
    )
    
    LOGGER.info("TFT attention analysis complete. Outputs saved to %s", output_dir)
    LOGGER.info("")


def example_unified_interpretation():
    """Demonstrate unified interpretation report generation."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 3: Unified Interpretation Report")
    LOGGER.info("=" * 80)
    
    # Load data
    data = pd.read_csv("data/processed/features.csv", parse_dates=['date'])
    data = data.set_index('date')
    
    target_col = 'oer_yoy'
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train XGBoost
    LOGGER.info("Training XGBoost for unified report...")
    xgb_model = XGBoostModel(name="xgboost", n_estimators=100)
    xgb_model.fit(X_train, y_train)
    
    # Generate comprehensive interpretation report
    output_dir = Path("outputs/interpretability/comprehensive")
    
    LOGGER.info("Generating comprehensive interpretation report...")
    artifacts = generate_interpretation_report(
        model=xgb_model.model_,
        X=X_test,
        y=y_test,
        model_name="xgboost",
        output_dir=output_dir,
        model_type="tree"
    )
    
    LOGGER.info("Generated %d interpretation artifacts:", len(artifacts))
    for artifact_name, artifact_path in artifacts.items():
        LOGGER.info("  - %s: %s", artifact_name, artifact_path)
    
    LOGGER.info("")


def example_performance_weighted_ensemble():
    """Demonstrate performance-weighted ensemble."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 4: Performance-Weighted Ensemble")
    LOGGER.info("=" * 80)
    
    # Simulate model forecasts and errors
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    
    # Forecasts from 3 models
    forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 100),
        'tft': np.random.normal(2.6, 0.25, 100),
        'var': np.random.normal(2.4, 0.4, 100)
    }, index=dates)
    
    # Historical errors (XGBoost best, VAR worst)
    errors = pd.DataFrame({
        'xgboost': np.random.normal(0, 0.15, 100),
        'tft': np.random.normal(0, 0.18, 100),
        'var': np.random.normal(0, 0.25, 100)
    }, index=dates)
    
    # Compute performance-weighted ensemble
    LOGGER.info("Computing performance-weighted ensemble (inverse RMSE)...")
    perf_ensemble = performance_weighted_average(
        forecasts=forecasts,
        errors=errors,
        method="inverse_rmse"
    )
    
    LOGGER.info("Performance-weighted ensemble created:")
    LOGGER.info("  Mean forecast: %.4f", perf_ensemble.mean())
    LOGGER.info("  Std forecast: %.4f", perf_ensemble.std())
    LOGGER.info("")


def example_bayesian_model_averaging():
    """Demonstrate Bayesian Model Averaging."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 5: Bayesian Model Averaging")
    LOGGER.info("=" * 80)
    
    # Simulate model forecasts and errors
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    
    forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 100),
        'tft': np.random.normal(2.6, 0.25, 100),
        'var': np.random.normal(2.4, 0.4, 100),
        'lasso': np.random.normal(2.55, 0.35, 100)
    }, index=dates)
    
    errors = pd.DataFrame({
        'xgboost': np.random.normal(0, 0.15, 100),
        'tft': np.random.normal(0, 0.18, 100),
        'var': np.random.normal(0, 0.25, 100),
        'lasso': np.random.normal(0, 0.20, 100)
    }, index=dates)
    
    # Compute BMA with uniform prior
    LOGGER.info("Computing Bayesian Model Averaging (uniform prior)...")
    bma_ensemble = bayesian_model_averaging(
        forecasts=forecasts,
        errors=errors,
        prior=None,  # Uniform
        temperature=1.0
    )
    
    LOGGER.info("BMA ensemble created:")
    LOGGER.info("  Mean forecast: %.4f", bma_ensemble.mean())
    LOGGER.info("")
    
    # Compute BMA with informative prior (higher weight to XGBoost)
    LOGGER.info("Computing BMA with informative prior (favor XGBoost)...")
    bma_ensemble_prior = bayesian_model_averaging(
        forecasts=forecasts,
        errors=errors,
        prior={'xgboost': 0.4, 'tft': 0.3, 'var': 0.15, 'lasso': 0.15},
        temperature=1.0
    )
    
    LOGGER.info("BMA ensemble (with prior) created:")
    LOGGER.info("  Mean forecast: %.4f", bma_ensemble_prior.mean())
    LOGGER.info("")


def example_stacking_ensemble():
    """Demonstrate stacking ensemble."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 6: Stacking Ensemble")
    LOGGER.info("=" * 80)
    
    # Simulate train and test forecasts
    train_dates = pd.date_range('2018-01-01', periods=80, freq='M')
    test_dates = pd.date_range('2024-09-01', periods=20, freq='M')
    
    train_forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 80),
        'tft': np.random.normal(2.6, 0.25, 80),
        'var': np.random.normal(2.4, 0.4, 80)
    }, index=train_dates)
    
    test_forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 20),
        'tft': np.random.normal(2.6, 0.25, 20),
        'var': np.random.normal(2.4, 0.4, 20)
    }, index=test_dates)
    
    # Simulate actuals
    train_actuals = pd.Series(
        np.random.normal(2.5, 0.2, 80),
        index=train_dates
    )
    
    # Ridge stacking
    LOGGER.info("Computing stacking ensemble (Ridge meta-model)...")
    stacked_ridge = stacking_ensemble(
        train_forecasts=train_forecasts,
        train_actuals=train_actuals,
        test_forecasts=test_forecasts,
        meta_model="ridge",
        alpha=1.0
    )
    
    LOGGER.info("Stacked forecast (Ridge):")
    LOGGER.info("  Mean: %.4f", stacked_ridge.mean())
    LOGGER.info("")
    
    # Random forest stacking
    LOGGER.info("Computing stacking ensemble (Random Forest meta-model)...")
    stacked_rf = stacking_ensemble(
        train_forecasts=train_forecasts,
        train_actuals=train_actuals,
        test_forecasts=test_forecasts,
        meta_model="rf",
        n_estimators=50,
        max_depth=3
    )
    
    LOGGER.info("Stacked forecast (RF):")
    LOGGER.info("  Mean: %.4f", stacked_rf.mean())
    LOGGER.info("")


def example_optimal_ensemble():
    """Demonstrate automatic ensemble method selection."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 7: Optimal Ensemble (Auto-Selection)")
    LOGGER.info("=" * 80)
    
    # Simulate train and test forecasts
    train_dates = pd.date_range('2018-01-01', periods=80, freq='M')
    test_dates = pd.date_range('2024-09-01', periods=20, freq='M')
    
    train_forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 80),
        'tft': np.random.normal(2.6, 0.25, 80),
        'var': np.random.normal(2.4, 0.4, 80)
    }, index=train_dates)
    
    test_forecasts = pd.DataFrame({
        'xgboost': np.random.normal(2.5, 0.3, 20),
        'tft': np.random.normal(2.6, 0.25, 20),
        'var': np.random.normal(2.4, 0.4, 20)
    }, index=test_dates)
    
    train_actuals = pd.Series(
        np.random.normal(2.5, 0.2, 80),
        index=train_dates
    )
    
    # Let the system choose the best method
    LOGGER.info("Auto-selecting best ensemble method...")
    optimal = optimal_ensemble(
        train_forecasts=train_forecasts,
        train_actuals=train_actuals,
        test_forecasts=test_forecasts,
        method="auto",
        cv_folds=5
    )
    
    LOGGER.info("Optimal ensemble forecast:")
    LOGGER.info("  Mean: %.4f", optimal.mean())
    LOGGER.info("")


def example_comparison_report():
    """Demonstrate comprehensive model comparison report."""
    LOGGER.info("=" * 80)
    LOGGER.info("EXAMPLE 8: Model Comparison Report")
    LOGGER.info("=" * 80)
    
    # Simulate backtest results for multiple models
    dates = pd.date_range('2020-01-01', periods=50, freq='M')
    
    backtest_results = {
        'xgboost': pd.DataFrame({
            'timestamp': dates,
            'y_true': np.random.normal(2.5, 0.2, 50),
            'y_pred': np.random.normal(2.5, 0.3, 50),
            'model': 'xgboost'
        }),
        'tft': pd.DataFrame({
            'timestamp': dates,
            'y_true': np.random.normal(2.5, 0.2, 50),
            'y_pred': np.random.normal(2.6, 0.25, 50),
            'model': 'tft'
        }),
        'var': pd.DataFrame({
            'timestamp': dates,
            'y_true': np.random.normal(2.5, 0.2, 50),
            'y_pred': np.random.normal(2.4, 0.4, 50),
            'model': 'var'
        })
    }
    
    # Add metrics
    for model_name, df in backtest_results.items():
        df['rmse'] = np.sqrt((df['y_true'] - df['y_pred']) ** 2)
        df['mae'] = np.abs(df['y_true'] - df['y_pred'])
        df['mape'] = np.abs((df['y_true'] - df['y_pred']) / df['y_true']) * 100
    
    # Create comparison report
    output_dir = Path("outputs/interpretability/comparison")
    
    LOGGER.info("Creating model comparison report...")
    report_path = create_model_comparison_report(
        backtest_results=backtest_results,
        output_dir=output_dir,
        include_interpretations=True
    )
    
    LOGGER.info("Comparison report saved to: %s", report_path)
    LOGGER.info("")


if __name__ == "__main__":
    LOGGER.info("OER Forecasting: Interpretability and Ensemble Examples")
    LOGGER.info("")
    
    # Run examples (comment out as needed)
    try:
        example_shap_interpretation()
    except Exception as e:
        LOGGER.error("SHAP example failed: %s", e)
    
    try:
        example_tft_attention()
    except Exception as e:
        LOGGER.error("TFT attention example failed: %s", e)
    
    try:
        example_unified_interpretation()
    except Exception as e:
        LOGGER.error("Unified interpretation example failed: %s", e)
    
    try:
        example_performance_weighted_ensemble()
    except Exception as e:
        LOGGER.error("Performance-weighted example failed: %s", e)
    
    try:
        example_bayesian_model_averaging()
    except Exception as e:
        LOGGER.error("BMA example failed: %s", e)
    
    try:
        example_stacking_ensemble()
    except Exception as e:
        LOGGER.error("Stacking example failed: %s", e)
    
    try:
        example_optimal_ensemble()
    except Exception as e:
        LOGGER.error("Optimal ensemble example failed: %s", e)
    
    try:
        example_comparison_report()
    except Exception as e:
        LOGGER.error("Comparison report example failed: %s", e)
    
    LOGGER.info("=" * 80)
    LOGGER.info("All examples complete!")
    LOGGER.info("=" * 80)
