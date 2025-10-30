"""
Complete End-to-End OER Forecasting Workflow Example

This script demonstrates the full pipeline from data ingestion to
forecast generation and interpretation, showcasing all major features.

Usage:
    python examples/complete_workflow.py
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("="*80)
print("OER FORECASTING - COMPLETE WORKFLOW DEMONSTRATION")
print("="*80)

# ==============================================================================
# STEP 1: CONFIGURATION
# ==============================================================================
print("\n[Step 1/7] Loading Configuration...")

from oer_model.config import AppConfig

config = AppConfig.from_yaml('config/config_enhanced.yaml')
print(f"✓ Loaded configuration: {config.project.get('name', 'oer_forecasting')}")
print(f"  - Data sources: FRED={config.data_sources.get('fred', {}).get('enabled', False)}, "
      f"Bloomberg={config.data_sources.get('bloomberg', {}).get('enabled', False)}")

# ==============================================================================
# STEP 2: DATA INGESTION
# ==============================================================================
print("\n[Step 2/7] Fetching Data...")

from oer_model.data.pipeline import collect_raw_data

try:
    raw_data = collect_raw_data(config)
    print(f"✓ Fetched {len(raw_data)} months of data")
    print(f"  - Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
    print(f"  - Series: {', '.join(raw_data.columns[:5])}{'...' if len(raw_data.columns) > 5 else ''}")
except Exception as e:
    print(f"✗ Data fetch failed: {e}")
    print("  NOTE: If Bloomberg BQL not available, ensure manual datasets are in data/external/")
    raw_data = None

# ==============================================================================
# STEP 3: FEATURE ENGINEERING
# ==============================================================================
print("\n[Step 3/7] Engineering Features...")

if raw_data is not None:
    from oer_model.features.engineering import build_feature_matrix
    from oer_model.features.panel_engineering import build_panel_aware_features
    
    # Standard features
    features = build_feature_matrix(config, raw_data)
    print(f"✓ Created {len(features.columns)} standard features")
    
    # Panel-aware features (if enabled)
    panel_config = config.features.get('panel_features', {})
    if panel_config.get('enabled', False):
        print("  - Adding panel-aware features (6-panel BLS structure)...")
        market_indicators = panel_config.get('market_indicators', [])
        market_indicators = [col for col in market_indicators if col in features.columns]
        
        if market_indicators:
            oer_col = panel_config.get('oer_column', 'oer_cpi')
            features = build_panel_aware_features(
                features,
                market_indicators=market_indicators,
                oer_column=oer_col,
                n_panels=panel_config.get('n_panels', 6)
            )
            print(f"✓ Added panel features. Total: {len(features.columns)} features")
    
    # Save processed features
    from oer_model.utils.io import write_dataframe
    output_path = config.processed_dir / "features_processed.csv"
    write_dataframe(features, output_path)
    print(f"✓ Saved features to {output_path}")
else:
    print("⊘ Skipping feature engineering (no raw data)")
    features = None

# ==============================================================================
# STEP 4: EXPLORATORY ANALYSIS
# ==============================================================================
print("\n[Step 4/7] Exploratory Analysis...")

if features is not None:
    target_col = config.features.get('target_column', 'oer_yoy')
    
    if target_col in features.columns:
        # Summary statistics
        print(f"\nTarget Variable: {target_col}")
        print(f"  - Mean: {features[target_col].mean():.2f}%")
        print(f"  - Std: {features[target_col].std():.2f}%")
        print(f"  - Min: {features[target_col].min():.2f}%")
        print(f"  - Max: {features[target_col].max():.2f}%")
        
        # Plot target variable
        fig, ax = plt.subplots(figsize=(14, 5))
        features[target_col].plot(ax=ax, linewidth=2, color='steelblue')
        ax.axhline(features[target_col].mean(), color='red', linestyle='--', 
                   label=f'Mean ({features[target_col].mean():.2f}%)', alpha=0.7)
        ax.set_title('OER Year-over-Year Change (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('YoY Change (%)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = config.artifacts_dir / "exploratory_oer_yoy.png"
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")
        plt.close()
    
    # Feature correlation with target
    if target_col in features.columns:
        correlations = features.corr()[target_col].sort_values(ascending=False)
        print(f"\nTop 10 Features Correlated with {target_col}:")
        for i, (feat, corr) in enumerate(correlations.head(11).items(), 1):
            if feat != target_col:
                print(f"  {i}. {feat}: {corr:.3f}")

else:
    print("⊘ Skipping exploratory analysis (no features)")

# ==============================================================================
# STEP 5: MODEL TRAINING & BACKTESTING
# ==============================================================================
print("\n[Step 5/7] Training Models and Running Backtests...")

if features is not None:
    from oer_model.evaluation.backtest import run_backtest, BacktestConfig
    from oer_model.models.baselines import LassoModel
    from oer_model.models.xgboost import XGBoostModel
    
    target_col = config.features.get('target_column', 'oer_yoy')
    
    if target_col not in features.columns:
        print(f"✗ Target column '{target_col}' not found in features")
    else:
        # Prepare data
        features_clean = features.dropna()
        X = features_clean.drop(columns=[target_col])
        y = features_clean[target_col]
        
        print(f"  - Training samples: {len(X)}")
        print(f"  - Features: {len(X.columns)}")
        
        # Backtest configuration
        backtest_cfg_dict = config.backtest or {}
        backtest_cfg = BacktestConfig(
            train_window=backtest_cfg_dict.get('train_window', 120),
            test_window=backtest_cfg_dict.get('test_window', 6),
            step=backtest_cfg_dict.get('step', 3)
        )
        
        results = {}
        
        # Model 1: LASSO
        print("\n  [Model 1: LASSO Regression]")
        try:
            def lasso_factory():
                return LassoModel(name='lasso', cv=5, max_iter=5000)
            
            lasso_results = run_backtest(features_clean, target_col, lasso_factory, backtest_cfg)
            results['lasso'] = lasso_results
            
            rmse = lasso_results['rmse'].iloc[0]
            mae = lasso_results['mae'].iloc[0]
            print(f"  ✓ LASSO backtest complete. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            print(f"  ✗ LASSO backtest failed: {e}")
        
        # Model 2: XGBoost
        print("\n  [Model 2: XGBoost]")
        try:
            def xgb_factory():
                return XGBoostModel(
                    name='xgboost',
                    params={
                        'n_estimators': 300,  # Reduced for speed
                        'learning_rate': 0.05,
                        'max_depth': 4
                    }
                )
            
            xgb_results = run_backtest(features_clean, target_col, xgb_factory, backtest_cfg)
            results['xgboost'] = xgb_results
            
            rmse = xgb_results['rmse'].iloc[0]
            mae = xgb_results['mae'].iloc[0]
            print(f"  ✓ XGBoost backtest complete. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            print(f"  ✗ XGBoost backtest failed: {e}")
        
        # Save results
        if results:
            for model_name, df in results.items():
                output_path = config.artifacts_dir / f"backtest_{model_name}.csv"
                from oer_model.utils.io import write_dataframe
                write_dataframe(df, output_path)
                print(f"  ✓ Saved {model_name} results to {output_path}")
else:
    print("⊘ Skipping model training (no features)")
    results = {}

# ==============================================================================
# STEP 6: MODEL INTERPRETATION
# ==============================================================================
print("\n[Step 6/7] Model Interpretation...")

if features is not None and results:
    target_col = config.features.get('target_column', 'oer_yoy')
    features_clean = features.dropna()
    X = features_clean.drop(columns=[target_col])
    y = features_clean[target_col]
    
    # Train final XGBoost for interpretation
    print("\n  [XGBoost Feature Importance via SHAP]")
    try:
        import shap
        
        xgb_final = XGBoostModel(
            params={
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 4
            }
        )
        xgb_final.fit(X, y)
        
        # Calculate SHAP values (use subset for speed)
        explainer = shap.TreeExplainer(xgb_final.estimator)
        shap_values = explainer.shap_values(X.iloc[-100:])  # Last 100 samples
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X.iloc[-100:], max_display=15, show=False)
        plt.title('XGBoost Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        shap_path = config.artifacts_dir / "xgboost_shap_summary.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved SHAP summary plot to {shap_path}")
        plt.close()
        
    except ImportError:
        print("  ⊘ SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"  ✗ SHAP interpretation failed: {e}")

else:
    print("⊘ Skipping interpretation (no models trained)")

# ==============================================================================
# STEP 7: FORECAST GENERATION
# ==============================================================================
print("\n[Step 7/7] Generating Forecasts...")

if features is not None:
    target_col = config.features.get('target_column', 'oer_yoy')
    features_clean = features.dropna()
    X = features_clean.drop(columns=[target_col])
    y = features_clean[target_col]
    
    # Train final models on full dataset
    print("\n  Training final models on complete dataset...")
    
    final_models = {}
    
    # LASSO
    try:
        lasso_final = LassoModel(name='lasso', cv=5, max_iter=5000)
        lasso_final.fit(X, y)
        final_models['lasso'] = lasso_final
        print("  ✓ LASSO trained")
    except Exception as e:
        print(f"  ✗ LASSO training failed: {e}")
    
    # XGBoost
    try:
        xgb_final = XGBoostModel(
            params={'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 4}
        )
        xgb_final.fit(X, y)
        final_models['xgboost'] = xgb_final
        print("  ✓ XGBoost trained")
    except Exception as e:
        print(f"  ✗ XGBoost training failed: {e}")
    
    # Generate forecasts
    if final_models:
        print("\n  Generating 12-month ahead forecasts...")
        
        # Use last observation as base for forecasting
        # Note: In production, you'd properly construct future features
        X_forecast = X.tail(12)  # Simplified: use last 12 months as proxy
        
        forecast_df = pd.DataFrame(index=pd.date_range(
            features_clean.index[-1] + pd.offsets.MonthEnd(),
            periods=12,
            freq='M'
        ))
        
        for model_name, model in final_models.items():
            try:
                preds = model.predict(X_forecast)
                forecast_df[model_name] = preds
                print(f"  ✓ {model_name} forecast generated")
            except Exception as e:
                print(f"  ✗ {model_name} forecast failed: {e}")
        
        # Save forecasts
        forecast_path = config.artifacts_dir / "latest_forecasts.csv"
        from oer_model.utils.io import write_dataframe
        write_dataframe(forecast_df, forecast_path)
        print(f"  ✓ Saved forecasts to {forecast_path}")
        
        # Plot forecasts
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        features_clean[target_col].iloc[-36:].plot(
            ax=ax, label='Historical OER', linewidth=2, color='black'
        )
        
        # Forecasts
        colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
        for i, col in enumerate(forecast_df.columns):
            forecast_df[col].plot(
                ax=ax, label=f'{col} forecast', 
                linewidth=2, linestyle='--', marker='o',
                color=colors[i % len(colors)]
            )
        
        ax.axvline(features_clean.index[-1], color='gray', linestyle=':', 
                   label='Forecast Start', alpha=0.7)
        ax.set_title('OER Forecast: 12-Month Ahead Projections', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('OER YoY Change (%)')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        forecast_plot_path = config.artifacts_dir / "forecast_chart.png"
        plt.savefig(forecast_plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved forecast chart to {forecast_plot_path}")
        plt.close()
        
        # Print summary
        print("\n  Forecast Summary (12-month ahead):")
        print(f"    Consensus: {forecast_df.iloc[-1].mean():.2f}%")
        print(f"    Range: {forecast_df.iloc[-1].min():.2f}% to {forecast_df.iloc[-1].max():.2f}%")
        print(f"    Std Dev: {forecast_df.iloc[-1].std():.2f}%")

else:
    print("⊘ Skipping forecast generation (no features)")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("WORKFLOW COMPLETE")
print("="*80)
print("\nGenerated Artifacts:")
print(f"  1. Processed features: {config.processed_dir / 'features_processed.csv'}")
print(f"  2. Backtest results: {config.artifacts_dir / 'backtest_*.csv'}")
print(f"  3. Forecasts: {config.artifacts_dir / 'latest_forecasts.csv'}")
print(f"  4. Plots: {config.artifacts_dir}/*.png")

print("\nNext Steps:")
print("  1. Review forecast charts in artifacts/ directory")
print("  2. Examine backtest performance metrics")
print("  3. Launch dashboard: python scripts/dashboard.py")
print("  4. For TFT interpretability: enable TFT in config and re-run")

print("\n" + "="*80)
print("END OF DEMONSTRATION")
print("="*80 + "\n")
