# OER Forecasting Platform

This repository implements a fully modular forecasting stack for Owners' Equivalent Rent (OER), combining traditional econometric baselines with modern machine learning and deep learning architectures. The design mirrors the research roadmap captured in `Forecasting OER with ML_DL.txt`, emphasizing explainability, rigorous backtesting, and Bloomberg (BQNT) integration.

## Repository Layout

```
config/                # Project- and model-level configuration (YAML)
data/
  raw/                 # Persisted raw pulls from FRED/Bloomberg/manual
  interim/             # Optional intermediate artifacts
  processed/           # Feature matrix ready for modeling
  external/            # Manually sourced datasets (e.g., Zillow exports)
scripts/               # One-command entry points for each workflow stage
src/oer_model/
  cli.py               # Click-based command suite (fetch/build/backtest/train)
  config.py            # Typed configuration loader
  data/                # FRED/Bloomberg/manual ingestion pipelines
  features/            # Feature engineering recipes & custom transformers
  models/              # Baseline (LASSO/VAR), XGBoost, and TFT model wrappers
  evaluation/          # Rolling backtests, metrics, reporting helpers
  forecasting/         # Training orchestration, ensembles, forecast export
  viz/                 # Plotly Dash dashboard factory for diagnostics
artifacts/             # Backtest outputs, forecast panels for the dashboard
models/                # Persisted fitted models (.pkl)
```

## End-to-End Workflow

1. **Configure** – tailor `config/config.yaml` to reflect data availability and feature recipes (lag structure, YOY transforms, model toggles).
2. **Ingest Data** – pull FRED/Bloomberg series and merge manual Zillow exports.
3. **Engineer Features** – apply the configured transformation pipeline (YOY, lags, rolling statistics) to produce the modeling matrix.
4. **Backtest Models** – evaluate all enabled candidates with walk-forward cross-validation and persist diagnostics.
5. **Train Final Models** – refit selected models on the full sample and export serialized artifacts for deployment/BQNT.
6. **Forecast & Visualize** – generate current projections and launch the interactive dashboard for performance storytelling.

## Command Line Usage

All workflows are exposed via the `oer_model` CLI (see `src/oer_model/cli.py`). Example commands from the repository root:

```bash
# 1. Pull the latest configured datasets
python -m oer_model.cli fetch-data

# 2. Build the processed feature matrix
python -m oer_model.cli build-features

# 3. Run rolling backtests for every enabled model
python -m oer_model.cli backtest

# 4. Fit final models on the full sample and persist .pkl artifacts
python -m oer_model.cli train

# 5. Produce the latest horizon forecasts (default 12 months)
python -m oer_model.cli forecast --horizon 12

# 6. Launch the Plotly Dash dashboard on http://127.0.0.1:8050
python scripts/dashboard.py
```

The corresponding helper scripts in `scripts/` simply wrap these commands for easy scheduling or notebook integration.

> 💡 Tip: create and activate a virtual environment, then run `pip install -e .[xgboost,bloomberg,deep]` to install the core package plus optional extras needed for specific data sources or model classes.

## Data Notes & Manual Inputs

- Bloomberg tickers can be enabled by installing `xbbg`/`blpapi` and setting the `data.bloomberg` section in `config/config.yaml` to `enabled: true`.
- Zillow (ZORI, ZHVI) and other proprietary datasets should be exported to `data/external/` and referenced in the `data.manual.datasets` configuration table.
- All series are aligned to monthly frequency and transformed into year-over-year or lagged features to respect the documented structural delay in OER.

## Modeling Stack

- **Baselines**: LASSO regression and VAR provide interpretable benchmarks grounded in macroeconomic intuition.
- **Gradient Boosting**: XGBoost captures non-linear interactions while remaining fast to iterate; SHAP explainers can be layered on top (hooks in `evaluation/`).
- **Deep Learning**: Temporal Fusion Transformer (TFT) scaffolding is included with optional PyTorch Forecasting dependencies for attention-driven narrative insights.

Each model is independently configurable (hyperparameters, deployment toggle) via the `models.candidates` section of the config file. Backtest windows default to a 10-year training span with 6-month forecast horizons, but all values are user-adjustable.

## Dashboard

`scripts/dashboard.py` spins up a Plotly Dash application that:

- Plots multi-model forecast paths versus actuals.
- Renders rolling-origin backtest comparisons.
- Surfaces scorecards (RMSE/MAE/MAPE) for quick executive review.

The dashboard consumes CSV artifacts generated during the backtest and forecast stages, making it simple to publish refreshed visuals after each data release.

## Next Steps

- Wire SHAP value computation for tree-based models and attention heatmaps for TFT into `evaluation/reporting.py`.
- Extend the ensemble module with Bayesian or performance-weighted averaging.
- Integrate Bloomberg BQNT export scripts to automate dashboard publishing directly within the terminal environment.