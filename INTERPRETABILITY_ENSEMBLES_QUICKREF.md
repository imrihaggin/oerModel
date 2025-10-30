# Interpretability & Ensembles Quick Reference

## SHAP for Tree Models

```python
from src.oer_model.evaluation.reporting import (
    compute_shap_values, plot_shap_summary, plot_shap_waterfall
)

# Compute
shap_result = compute_shap_values(model.model_, X_test, model_type="tree")

# Plots
plot_shap_summary(shap_result, "summary.png", plot_type="dot")
plot_shap_waterfall(shap_result, -1, "waterfall.png")
```

**Outputs:**
- `dot`: Feature importance beeswarm
- `bar`: Average |SHAP| values
- `waterfall`: Single prediction breakdown

---

## TFT Attention

```python
from src.oer_model.evaluation.reporting import (
    extract_tft_attention, plot_tft_attention_heatmap
)

# Extract
attn = extract_tft_attention(tft_model, X_test, sample_idx=-1)

# Plot
plot_tft_attention_heatmap(attn, "attention.png")
```

**Shows:** Which historical periods drive each forecast horizon

---

## Unified Report

```python
from src.oer_model.evaluation.reporting import generate_interpretation_report

artifacts = generate_interpretation_report(
    model=model,
    X=X_test,
    y=y_test,
    model_name="xgb",
    output_dir=Path("outputs"),
    model_type="auto"  # auto-detects tree vs tft
)
```

**Auto-generates:** All relevant plots + CSV for model type

---

## Ensemble Methods

### 1. Performance Weighted
```python
from src.oer_model.forecasting.ensembles import performance_weighted_average

ens = performance_weighted_average(forecasts_df, errors_df, "inverse_rmse")
```
**Use when:** Clear performance differences

### 2. Bayesian Model Averaging
```python
from src.oer_model.forecasting.ensembles import bayesian_model_averaging

ens = bayesian_model_averaging(
    forecasts_df, 
    errors_df,
    prior={'xgb': 0.4, 'tft': 0.3, 'var': 0.3}
)
```
**Use when:** Want probabilistic weighting + priors

### 3. Stacking
```python
from src.oer_model.forecasting.ensembles import stacking_ensemble

ens = stacking_ensemble(
    train_fc, train_y, test_fc,
    meta_model="ridge", alpha=1.0
)
```
**Use when:** Have training data, want learned weights

### 4. Dynamic Weighted
```python
from src.oer_model.forecasting.ensembles import dynamic_weighted_ensemble

ens = dynamic_weighted_ensemble(
    forecasts_df, errors_df,
    window=12, method="inverse_rmse"
)
```
**Use when:** Performance changes over time

### 5. Robust
```python
from src.oer_model.forecasting.ensembles import robust_ensemble

ens = robust_ensemble(forecasts_df, method="median")
```
**Use when:** Worried about outliers

### 6. Optimal (Auto-Select)
```python
from src.oer_model.forecasting.ensembles import optimal_ensemble

ens = optimal_ensemble(train_fc, train_y, test_fc, method="auto")
```
**Use when:** Unsure which method to use

---

## Complete Workflow

```python
# 1. Train
xgb = XGBoostModel(name="xgb")
xgb.fit(X_train, y_train)

# 2. Interpret
generate_interpretation_report(
    xgb.model_, X_test, y_test, "xgb", Path("outputs")
)

# 3. Ensemble
forecasts = pd.DataFrame({'xgb': xgb.predict(X_test), ...})
errors = pd.DataFrame({'xgb': y_test - forecasts['xgb'], ...})
ensemble = bayesian_model_averaging(forecasts, errors)
```

---

## Files Created

✅ `src/oer_model/evaluation/reporting.py` (700+ lines)
✅ `src/oer_model/forecasting/ensembles.py` (550+ lines)
✅ `examples/interpretability_and_ensembles.py` (350 lines)
✅ `INTERPRETABILITY_ENSEMBLES_GUIDE.md` (complete guide)
✅ `INTEGRATION_COMPLETE.md` (summary)
✅ `INTERPRETABILITY_ENSEMBLES_QUICKREF.md` (this file)

---

## Key Decision Matrix

| Scenario | SHAP | Attention | Ensemble Method |
|----------|------|-----------|-----------------|
| Tree model + need feature importance | ✅ | ❌ | Performance/BMA |
| TFT + need temporal understanding | ❌ | ✅ | BMA/Stacking |
| Multiple models, clear best performer | - | - | Performance |
| Uncertain which to use | - | - | Optimal (auto) |
| Models have similar performance | - | - | Equal/Median |
| Need probabilistic interpretation | - | - | BMA |
| Large training set available | - | - | Stacking |
| Performance varies over time | - | - | Dynamic |

---

## Installation

```bash
pip install -r requirements.txt
```

Includes: `shap>=0.42`, `scikit-learn>=1.2`, `matplotlib>=3.7`, `seaborn>=0.12`

---

## Examples

```bash
python examples/interpretability_and_ensembles.py
```

Runs 8 comprehensive examples demonstrating all features.

---

## References

📖 Full Guide: `INTERPRETABILITY_ENSEMBLES_GUIDE.md`  
📝 Summary: `INTEGRATION_COMPLETE.md`  
💻 Examples: `examples/interpretability_and_ensembles.py`  
📊 Code: `src/oer_model/evaluation/reporting.py` + `src/oer_model/forecasting/ensembles.py`
