# Quick Start Guide - OER Forecasting Platform

## 🚀 Get Started in 5 Minutes

This guide will get you from zero to forecasting in minimal time.

---

## Prerequisites

- Python 3.8 or higher
- 30 minutes for first-time setup (data download + model training)
- For Bloomberg Terminal users: BQL access

---

## Installation

```bash
# 1. Clone and enter directory
git clone <your-repo-url> oerModel
cd oerModel

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# Optional: Install deep learning dependencies for TFT
pip install torch pytorch-lightning pytorch-forecasting
```

---

## Option A: Quick Demo (No Bloomberg, Uses FRED Public Data)

### Step 1: Configure
```bash
# Use the enhanced configuration
cp config/config_enhanced.yaml config/config.yaml

# Verify FRED is enabled (it should be by default)
# bloomberg.enabled: false  ← Should be false
# fred.enabled: true        ← Should be true
```

### Step 2: Run Complete Workflow
```bash
python examples/complete_workflow.py
```

This will:
- ✅ Fetch data from FRED
- ✅ Engineer features (including panel-aware)
- ✅ Train LASSO and XGBoost models
- ✅ Run backtests
- ✅ Generate forecasts
- ✅ Create visualizations

**Output**: All results in `artifacts/` directory

### Step 3: View Results
```bash
# Launch dashboard
python scripts/dashboard.py --port 8050
```

Open browser to: `http://127.0.0.1:8050`

---

## Option B: Bloomberg Terminal (BQNT) Setup

### Step 1: Enable Bloomberg in Config
```yaml
# config/config.yaml
data:
  bloomberg:
    enabled: true  # ← Change this
    series:
      - ticker: "CPIQOEPS Index"
        alias: oer_cpi_bloomberg
      - ticker: "ZRIOAYOY Index"
        alias: zori_yoy
      # ... (rest of tickers)
```

### Step 2: Fetch Bloomberg Data
```bash
python make_dataset.py data/raw/bloomberg_data.csv --start-date 2010-01-01
```

### Step 3: Run Pipeline
```bash
# Build features
python -m oer_model.cli build-features

# Train models
python -m oer_model.cli train

# Generate forecasts
python -m oer_model.cli forecast --horizon 12

# Launch dashboard
python scripts/dashboard.py
```

---

## Key Files to Check After Running

```
artifacts/
├── backtest_lasso.csv          # LASSO rolling backtest results
├── backtest_xgboost.csv        # XGBoost rolling backtest results
├── latest_forecasts.csv        # 12-month ahead forecasts
├── exploratory_oer_yoy.png     # Historical OER chart
├── forecast_chart.png          # Forecast visualization
└── xgboost_shap_summary.png    # Feature importance (if SHAP installed)

data/
├── processed/
│   └── features_processed.csv  # All engineered features ready for modeling

models/
├── lasso.pkl                   # Saved LASSO model
└── xgboost.pkl                 # Saved XGBoost model
```

---

## Understanding the Output

### Forecast File (`latest_forecasts.csv`)
```
date,lasso,xgboost
2025-11-30,4.23,4.18
2025-12-31,4.28,4.22
2026-01-31,4.31,4.25
...
```

**Interpretation**: Models predict OER will be growing at ~4.2% YoY in November 2025.

### Backtest Results (`backtest_xgboost.csv`)
```
timestamp,y_true,y_pred,rmse,mae,model
2023-01-31,7.89,7.65,0.35,0.25,xgboost
2023-02-28,8.13,7.92,0.35,0.25,xgboost
...
```

**Key Metrics**:
- **RMSE < 0.5**: Excellent for OER (typical range 0.3-0.6)
- **MAE < 0.3**: Good for policy-relevant accuracy

### Features (`features_processed.csv`)

Key columns to understand:
- `oer_yoy`: **Target variable** (OER year-over-year % change)
- `zori_yoy_lag12m`: ZORI lagged 12 months (primary predictor)
- `zhvi_yoy_lag16m`: ZHVI lagged 16 months (secondary predictor)
- `unemployment_lag3m`: Unemployment rate lagged 3 months
- `*_panel_ma`: Panel-weighted moving averages (BLS view)
- `*_oer_spread`: Market vs OER divergence

---

## Customization

### Add Your Own Predictors

Edit `config/config.yaml`:

```yaml
data:
  fred:
    series:
      # Add new FRED series
      - series_id: YOUR_SERIES_ID
        alias: your_feature_name
        description: "Description"

features:
  recipes:
    # Add transformation recipe
    - source: your_feature_name
      operations:
        - type: pct_change
          periods: 12
          alias: your_feature_yoy
        - type: lag
          periods: 6
          alias: your_feature_yoy_lag6m
```

Then re-run:
```bash
python -m oer_model.cli fetch-data
python -m oer_model.cli build-features
```

### Add New Models

1. Create model class in `src/oer_model/models/`:
```python
from .base import ForecastModel

class MyModel(ForecastModel):
    def fit(self, X, y):
        # Your code
        return self
    
    def predict(self, X):
        # Your code
        return predictions
```

2. Add to config:
```yaml
models:
  candidates:
    my_model:
      enabled: true
      deploy: true
      class: oer_model.models.my_models.MyModel
      params:
        name: my_model
        # your parameters
```

3. Run backtest:
```bash
python -m oer_model.cli backtest
```

Model automatically appears in dashboard!

---

## Troubleshooting

### "No module named 'bql'"
**Solution**: BQL only available in Bloomberg Terminal. Use FRED data instead:
```yaml
data:
  bloomberg:
    enabled: false
  fred:
    enabled: true
```

### "pytorch-forecasting not found"
**Solution**: TFT requires deep learning libs:
```bash
pip install torch pytorch-lightning pytorch-forecasting
```

Or disable TFT in config:
```yaml
models:
  candidates:
    tft:
      enabled: false
```

### "Feature column not found"
**Solution**: Feature engineering might have failed. Check for NaN values:
```python
import pandas as pd
df = pd.read_csv('data/processed/features_processed.csv')
print(df.isnull().sum())  # Check missing values
```

### "Backtest produced no splits"
**Solution**: Not enough data for backtest window. Reduce `train_window`:
```yaml
backtest:
  train_window: 60  # Default is 120 (10 years)
```

---

## Next Steps

### 1. Explore the Dashboard
- **Overview Tab**: See all model forecasts
- **Performance Tab**: Compare backtest results
- **Interpretation Tab**: Understand what drives forecasts (if TFT enabled)

### 2. Examine Feature Engineering
```bash
jupyter notebook
# Open: notebooks/01_feature_exploration.ipynb
```

### 3. Deep Dive into TFT Interpretability
```bash
# Enable TFT in config, then:
python -m oer_model.cli train

# Check interpretation artifacts:
ls artifacts/interpretation/
# -> variable_importance.png
# -> attention_heatmap.png
# -> explanations.json
```

### 4. Read Full Documentation
- `README_ENHANCED.md`: Comprehensive guide (1,200+ lines)
- `IMPROVEMENTS_SUMMARY.md`: What's new in v2.0
- Docstrings in code: All functions documented

---

## Common Workflows

### Weekly Forecast Update
```bash
# 1. Fetch latest data
python -m oer_model.cli fetch-data

# 2. Rebuild features
python -m oer_model.cli build-features

# 3. Generate new forecasts (uses pre-trained models)
python -m oer_model.cli forecast --horizon 12

# 4. Check dashboard
python scripts/dashboard.py
```

### Quarterly Model Retraining
```bash
# Full retrain on updated data
python -m oer_model.cli fetch-data
python -m oer_model.cli build-features
python -m oer_model.cli train  # Refits all models
python -m oer_model.cli forecast
```

### Research: Test New Feature
```bash
# 1. Add feature to config (see "Add Your Own Predictors" above)

# 2. Rebuild and backtest
python -m oer_model.cli build-features
python -m oer_model.cli backtest

# 3. Compare metrics
cat artifacts/summary_xgboost.csv  # Check RMSE before/after
```

---

## Performance Expectations

**First Run** (with data download):
- Data fetch: 2-5 minutes
- Feature engineering: 30 seconds
- LASSO training: 10 seconds
- XGBoost training: 1-2 minutes
- XGBoost backtest: 3-5 minutes
- TFT training (if enabled): 10-20 minutes

**Subsequent Runs** (data cached):
- Feature engineering: 10 seconds
- Model training: 1-2 minutes total
- Forecasting: < 1 second

---

## Support

- **Issues**: Open GitHub issue
- **Questions**: Check `README_ENHANCED.md` first
- **Examples**: See `examples/complete_workflow.py`

---

## Quick Reference Card

```bash
# SETUP
pip install -r requirements.txt && pip install -e .

# FETCH DATA
python -m oer_model.cli fetch-data                    # FRED
python make_dataset.py data/raw/bbg.csv               # Bloomberg

# BUILD & TRAIN
python -m oer_model.cli build-features                # Create features
python -m oer_model.cli backtest                      # Evaluate models
python -m oer_model.cli train                         # Fit final models

# FORECAST & VISUALIZE  
python -m oer_model.cli forecast --horizon 12         # Generate forecasts
python scripts/dashboard.py --port 8050               # Launch dashboard

# ALL-IN-ONE
python examples/complete_workflow.py                  # Full demo
```

---

**Ready to forecast!** 🚀

Run `python examples/complete_workflow.py` to see everything in action.
