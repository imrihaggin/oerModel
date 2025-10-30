# Interpretability & Advanced Ensembles Guide

## Overview

This guide covers the newly integrated interpretability and advanced ensemble functionality in the OER forecasting repository. These features enable deep understanding of model predictions and sophisticated forecast combination strategies.

## Table of Contents

1. [SHAP Integration](#shap-integration)
2. [TFT Attention](#tft-attention)
3. [Advanced Ensemble Methods](#advanced-ensemble-methods)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

---

## SHAP Integration

### What is SHAP?

SHAP (SHapley Additive exPlanations) provides a unified framework for interpreting tree-based models by quantifying each feature's contribution to individual predictions.

### Available Functions

Located in `src/oer_model/evaluation/reporting.py`:

#### `compute_shap_values()`
Compute SHAP values for a fitted model.

```python
from src.oer_model.evaluation.reporting import compute_shap_values

shap_result = compute_shap_values(
    model=xgb_model.model_,      # Fitted XGBoost model
    X=X_test,                     # Feature matrix
    model_type="tree",            # 'tree', 'linear', or 'kernel'
    max_samples=500               # Subsample for efficiency
)
```

**Returns:**
- `shap_values`: SHAP value matrix (n_samples x n_features)
- `base_value`: Expected model output
- `feature_names`: List of feature names
- `data`: Sample data used
- `explainer`: SHAP explainer object

#### `plot_shap_summary()`
Create SHAP summary plot showing feature importance.

```python
from src.oer_model.evaluation.reporting import plot_shap_summary

plot_shap_summary(
    shap_result=shap_result,
    output_path=Path("outputs/shap_summary.png"),
    max_display=15,              # Top N features
    plot_type="dot"              # 'dot', 'bar', or 'violin'
)
```

**Plot Types:**
- `dot` (beeswarm): Shows feature values and SHAP impacts
- `bar`: Average absolute SHAP values (feature importance)
- `violin`: Distribution of SHAP values per feature

#### `plot_shap_waterfall()`
Show feature contributions for a single prediction.

```python
from src.oer_model.evaluation.reporting import plot_shap_waterfall

plot_shap_waterfall(
    shap_result=shap_result,
    sample_idx=-1,               # Index of prediction to explain
    output_path=Path("outputs/shap_waterfall.png")
)
```

**Interpretation:**
- Base value: Average model prediction
- Red bars: Features pushing prediction higher
- Blue bars: Features pushing prediction lower
- Final value: Actual prediction

#### `plot_shap_dependence()`
Show how a feature's value affects predictions.

```python
from src.oer_model.evaluation.reporting import plot_shap_dependence

plot_shap_dependence(
    shap_result=shap_result,
    feature_name="shelter_cpi_yoy",
    interaction_feature="fed_funds_rate",  # Optional coloring
    output_path=Path("outputs/shap_dep.png")
)
```

---

## TFT Attention

### What is TFT Attention?

The Temporal Fusion Transformer uses multi-head attention to determine which historical time steps are most relevant for each forecast horizon. This provides temporal interpretability.

### Available Functions

Located in `src/oer_model/evaluation/reporting.py`:

#### `extract_tft_attention()`
Extract attention patterns from a fitted TFT model.

```python
from src.oer_model.evaluation.reporting import extract_tft_attention

attention_result = extract_tft_attention(
    tft_model=tft_model,         # Fitted InterpretableTFT instance
    X=X_test,                    # Feature matrix
    sample_idx=None              # None = last sample, or specific index
)
```

**Returns:**
- `attention`: Attention weight matrix (forecast_horizon x encoder_length)
- `time_labels`: Labels for historical time steps
- `aggregated`: Whether attention is aggregated across heads
- `model_name`: Name of the model

#### `plot_tft_attention_heatmap()`
Visualize attention patterns as a heatmap.

```python
from src.oer_model.evaluation.reporting import plot_tft_attention_heatmap

plot_tft_attention_heatmap(
    attention_result=attention_result,
    output_path=Path("outputs/attention_heatmap.png"),
    figsize=(14, 8),
    cmap='YlOrRd'               # Colormap
)
```

**Interpretation:**
- **Rows**: Forecast horizons (t+1, t+2, ..., t+H)
- **Columns**: Historical time steps (t-24, t-23, ..., t-1, t)
- **Colors**: Attention weights (brighter = more important)

#### `extract_tft_variable_importance()`
Get variable importance rankings from TFT.

```python
from src.oer_model.evaluation.reporting import extract_tft_variable_importance

importance_df = extract_tft_variable_importance(
    tft_model=tft_model,
    top_k=20                     # Number of top features
)
```

---

## Unified Interpretation

### `generate_interpretation_report()`

Auto-generates comprehensive interpretation artifacts for any model.

```python
from src.oer_model.evaluation.reporting import generate_interpretation_report

artifacts = generate_interpretation_report(
    model=model,                 # Fitted model (XGBoost or TFT)
    X=X_test,
    y=y_test,
    model_name="xgboost",
    output_dir=Path("outputs/interpretability"),
    model_type="auto"            # 'auto', 'tree', or 'tft'
)
```

**For Tree Models (XGBoost, LightGBM, RF):**
- SHAP summary plot (beeswarm)
- SHAP bar plot (feature importance)
- SHAP waterfall plot (latest prediction)
- SHAP dependence plots (top 3 features)

**For TFT Models:**
- Attention heatmap
- Variable importance CSV
- Variable importance plot
- Full interpretation report (if available)

### `create_model_comparison_report()`

Generate markdown report comparing all models with interpretations.

```python
from src.oer_model.evaluation.reporting import create_model_comparison_report

report_path = create_model_comparison_report(
    backtest_results=backtest_results,  # Dict[str, pd.DataFrame]
    output_dir=Path("outputs/comparison"),
    include_interpretations=True
)
```

---

## Advanced Ensemble Methods

All ensemble functions are in `src/oer_model/forecasting/ensembles.py`.

### 1. Equal Weight Average (Baseline)

Simple average of all forecasts.

```python
from src.oer_model.forecasting.ensembles import equal_weight_average

ensemble = equal_weight_average(forecasts)
```

**When to Use:**
- Baseline comparison
- Models have similar performance
- No historical error data available

---

### 2. Performance-Weighted Average

Weight models inversely by their historical errors.

```python
from src.oer_model.forecasting.ensembles import performance_weighted_average

ensemble = performance_weighted_average(
    forecasts=forecasts_df,      # DataFrame: columns = models
    errors=errors_df,            # DataFrame: historical errors
    method="inverse_rmse"        # 'inverse_rmse', 'inverse_mae', 'squared_inverse'
)
```

**Methods:**
- `inverse_rmse`: $w_i = \frac{1}{RMSE_i}$ (standard)
- `inverse_mae`: $w_i = \frac{1}{MAE_i}$ (robust to outliers)
- `squared_inverse`: $w_i = \frac{1}{RMSE_i^2}$ (penalizes poor models heavily)

**When to Use:**
- Clear performance differences between models
- Stable historical performance
- Want to downweight poor performers

---

### 3. Bayesian Model Averaging (BMA)

Weight models by posterior probability given the data.

```python
from src.oer_model.forecasting.ensembles import bayesian_model_averaging

ensemble = bayesian_model_averaging(
    forecasts=forecasts_df,
    errors=errors_df,
    prior=None,                  # Dict[str, float] or None (uniform)
    temperature=1.0              # Controls weight concentration
)
```

**Theory:**

$$P(model|data) \propto P(data|model) \cdot P(model)$$

Where:
- $P(model)$ = prior probability (uniform if None)
- $P(data|model)$ = likelihood (based on forecast errors)
- Uses softmax with temperature control

**When to Use:**
- Want probabilistic interpretation of model weights
- Have informative priors about model quality
- Need principled uncertainty quantification

**Prior Example:**
```python
# Favor XGBoost and TFT based on domain knowledge
prior = {
    'xgboost': 0.35,
    'tft': 0.35,
    'var': 0.15,
    'lasso': 0.15
}
```

---

### 4. Variance-Weighted Ensemble

Weight models inversely by their prediction uncertainty.

```python
from src.oer_model.forecasting.ensembles import variance_weighted_ensemble

ensemble = variance_weighted_ensemble(
    forecasts=forecasts_df,
    prediction_intervals=pi_df,  # Optional: prediction interval widths
    rolling_std=std_df,          # Optional: rolling forecast std
    window=12                    # Window for computing rolling std
)
```

**When to Use:**
- Models provide prediction intervals
- Want to downweight uncertain forecasts
- Forecast uncertainty varies over time

---

### 5. Stacking Ensemble

Use a meta-model to learn optimal combination weights.

```python
from src.oer_model.forecasting.ensembles import stacking_ensemble

ensemble = stacking_ensemble(
    train_forecasts=train_fc_df,
    train_actuals=train_y,
    test_forecasts=test_fc_df,
    meta_model="ridge",          # 'ridge', 'lasso', 'rf', 'custom'
    alpha=1.0                    # Regularization (for ridge/lasso)
)
```

**Meta-Model Options:**
- `ridge`: Ridge regression (L2 regularization, positive weights)
- `lasso`: Lasso regression (L1 regularization, sparse weights)
- `rf`: Random Forest (captures nonlinear combinations)
- `custom`: Provide your own model via `model=` kwarg

**When to Use:**
- Have sufficient training data
- Suspect nonlinear or interactive relationships
- Want data-driven weight learning

---

### 6. Dynamic Weighted Ensemble

Time-varying weights based on recent performance.

```python
from src.oer_model.forecasting.ensembles import dynamic_weighted_ensemble

ensemble = dynamic_weighted_ensemble(
    forecasts=forecasts_df,
    errors=errors_df,
    window=12,                   # Rolling window for weight computation
    method="inverse_rmse"
)
```

**When to Use:**
- Model performance changes over time
- Want to adapt to regime shifts
- Recent performance more informative than overall

---

### 7. Robust Ensemble

Less sensitive to outlier forecasts.

```python
from src.oer_model.forecasting.ensembles import robust_ensemble

ensemble = robust_ensemble(
    forecasts=forecasts_df,
    method="median",             # 'median', 'trimmed_mean', 'winsorized_mean'
    trim_pct=0.1                 # For trimmed/winsorized
)
```

**Methods:**
- `median`: Most robust (50th percentile)
- `trimmed_mean`: Remove extreme values from tails
- `winsorized_mean`: Replace extremes with percentile values

**When to Use:**
- Some models produce occasional outliers
- Want protection against extreme forecasts
- Forecast distribution is heavy-tailed

---

### 8. Optimal Ensemble (Auto-Selection)

Automatically select best ensemble method via cross-validation.

```python
from src.oer_model.forecasting.ensembles import optimal_ensemble

ensemble = optimal_ensemble(
    train_forecasts=train_fc_df,
    train_actuals=train_y,
    test_forecasts=test_fc_df,
    method="auto",               # 'auto' or specific method
    cv_folds=5
)
```

**Evaluated Methods:**
- Equal weight
- Median
- Stacking (Ridge)
- Stacking (Random Forest)

**When to Use:**
- Unsure which method to use
- Want data-driven method selection
- Computational cost not a concern

---

## Usage Examples

### Complete Workflow Example

```python
from pathlib import Path
import pandas as pd
from src.oer_model.models.xgboost import XGBoostModel
from src.oer_model.models.tft_interpretable import InterpretableTFT
from src.oer_model.evaluation.reporting import generate_interpretation_report
from src.oer_model.forecasting.ensembles import bayesian_model_averaging

# 1. Load data
data = pd.read_csv("data/processed/features.csv")
X_train, X_test, y_train, y_test = train_test_split(...)

# 2. Train models
xgb = XGBoostModel(name="xgb", n_estimators=100)
xgb.fit(X_train, y_train)

tft = InterpretableTFT(name="tft", max_epochs=50)
tft.fit(X_train, y_train)

# 3. Generate interpretations
generate_interpretation_report(
    model=xgb.model_,
    X=X_test,
    y=y_test,
    model_name="xgb",
    output_dir=Path("outputs/xgb_interpretation"),
    model_type="tree"
)

generate_interpretation_report(
    model=tft,
    X=X_test,
    y=y_test,
    model_name="tft",
    output_dir=Path("outputs/tft_interpretation"),
    model_type="tft"
)

# 4. Create ensemble
forecasts = pd.DataFrame({
    'xgb': xgb.predict(X_test),
    'tft': tft.predict(X_test)
})

errors = pd.DataFrame({
    'xgb': y_test - xgb.predict(X_test),
    'tft': y_test - tft.predict(X_test)
})

bma_forecast = bayesian_model_averaging(forecasts, errors)
```

---

## Best Practices

### SHAP Analysis

1. **Subsample for Speed**: Use `max_samples=500` for large datasets
2. **Focus on Top Features**: Set `max_display=15` for clarity
3. **Check Dependence Plots**: Understand feature interactions
4. **Waterfall for Stakeholders**: Best for explaining specific predictions

### TFT Attention

1. **Aggregate Across Heads**: Easier interpretation
2. **Focus on Key Horizons**: 1-month, 3-month, 6-month, 12-month
3. **Compare to Domain Knowledge**: Validate learned patterns
4. **Check Variable Importance**: Complements attention analysis

### Ensemble Selection

**Start Simple, Add Complexity:**
1. Equal weight (baseline)
2. Performance-weighted (if clear winners)
3. BMA (if want probabilistic interpretation)
4. Stacking (if have training data and time)

**Consider:**
- **Data availability**: Stacking requires train set
- **Computational cost**: Dynamic weighting is expensive
- **Interpretability**: Performance-weighted easiest to explain
- **Robustness**: Median best for outliers

### Production Deployment

1. **Save Interpretation Artifacts**: For audit trail
2. **Monitor Ensemble Weights**: Detect regime changes
3. **Refresh Weights Periodically**: Update with new data
4. **Document Model Selection**: Explain why certain methods chosen

---

## Configuration

Add to `config/config_enhanced.yaml`:

```yaml
interpretation:
  shap:
    enabled: true
    max_samples: 500
    plot_types: ['dot', 'bar', 'waterfall']
    max_display: 15
    
  tft_attention:
    enabled: true
    aggregate_heads: true
    key_horizons: [1, 3, 6, 12]
    
ensemble:
  method: "auto"  # 'auto', 'equal', 'performance', 'bma', 'stacking', etc.
  
  performance_weighted:
    method: "inverse_rmse"
    
  bma:
    temperature: 1.0
    prior: null  # or dict of priors
    
  stacking:
    meta_model: "ridge"
    alpha: 1.0
    
  dynamic:
    window: 12
    method: "inverse_rmse"
    
  robust:
    method: "median"
    trim_pct: 0.1
```

---

## Dependencies

Required packages (already in `requirements.txt`):

```txt
# Interpretability
shap>=0.42.0

# Ensemble meta-models
scikit-learn>=1.2.0

# Plotting
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## References

### SHAP
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
- Documentation: https://shap.readthedocs.io/

### TFT Attention
- Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." *International Journal of Forecasting*.

### Ensemble Methods
- Raftery, A. E., et al. (2005). "Using Bayesian Model Averaging to Calibrate Forecast Ensembles." *Monthly Weather Review*.
- Wolpert, D. H. (1992). "Stacked Generalization." *Neural Networks*.

---

## Support

For issues or questions:
1. Check `examples/interpretability_and_ensembles.py` for usage examples
2. Review logs in `outputs/logs/` for debugging
3. Consult SHAP documentation for advanced features
4. See `IMPROVEMENTS_SUMMARY.md` for architecture overview
