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
- Integrate Bloomberg 

I think it may still be salvageable—but I’d change the research question.

The weak version of the project is “aggregate LLM capex acceleration predicts the macro capex cycle.” As you’ve noticed, if the useful transformation is basically 1 – acceleration breadth, a lot of the apparent signal may just be cycle/base-effect information that simpler data already contains.

The stronger question is: what information exists inside the transcript-level cross section that disappears when you aggregate it? That is where the LLM extraction has a much better chance of producing genuine alpha.

I’d try these in roughly this order:

1. Unexpected acceleration rather than acceleration. Build an expected capex-acceleration probability from observable variables available before the call—industry, prior capex growth, sales growth, ISM, orders, stock returns, rates, prior-quarter language, etc. Then define
    \text{LLM surprise}_{i,t}=D_{i,t}-E[D_{i,t}\mid X_{t-1}].
    This is potentially much more interesting. “Company says capex is accelerating” is cyclical. “Company says capex is accelerating when the macro/financial data says it shouldn’t be” may be information.
2. Changes in language at the firm level. Instead of the level of acceleration breadth, calculate transitions: decelerating → stable → accelerating. In particular, I’d isolate first accelerations after several quarters of stable/decelerating language. Repeatedly saying “capex remains strong” probably contains little new information; the inflection could.
3. Dispersion and diffusion underneath the aggregate. Your thousands of observations give you something the macro series can’t. Measure cross-sectional dispersion, percentiles, skew, breadth within industries, and the number of industries turning simultaneously. For example, aggregate acceleration could be flat while machinery, semis and electrical equipment are suddenly accelerating and software/communications are decelerating. The aggregate destroys that information.
4. Supplier/customer propagation. This might be the highest-alpha use of the dataset. Don’t ask whether company i‘s capex language predicts aggregate capex. Ask whether capex acceleration among customers predicts future revenue/orders/margins for their suppliers. If hyperscalers suddenly accelerate data-center capex language, which equipment, electrical, cooling, construction, semiconductor and power suppliers subsequently receive estimate revisions? You’re converting textual information into an economic network signal.
5. Capex composition. “Capex acceleration” is probably too coarse. An LLM can distinguish why and what: capacity expansion, maintenance, automation/productivity, data centers/AI, reshoring, environmental/regulatory, new product, construction, equipment, IT/software, etc. Traditional macro data cannot give you that cleanly in real time. A capacity-expansion acceleration index could behave completely differently from maintenance capex.
6. Commitment/intention strength. Separate “considering/increasing/planning” from “approved/ordered/under construction.” You could construct an investment pipeline:
    \text{discussion}\rightarrow\text{budgeted}\rightarrow\text{ordered}\rightarrow\text{construction}\rightarrow\text{operational}.
    That has a much more natural lead structure than a generic acceleration classifier.
7. Guidance surprise. Compare the current transcript’s capex statement with the firm’s previous guidance. “Capex will be $8–9bn” isn’t interesting by itself; moving from $6bn to $8–9bn is. Even better, distinguish explicit numerical revisions from qualitative upgrades. That gives you a genuine event variable rather than a cyclical state variable.
8. Cross-sectional equity tests before macro tests. For each earnings date, freeze the information set and ask whether your signal predicts subsequent earnings revisions, revenue surprises, supplier returns, investment growth or analyst capex revisions over 1–12 months. Neutralize industry, size, momentum, value, profitability, beta, and contemporaneous earnings surprise. If the LLM signal has no incremental predictive content there, I’d be much less optimistic about extracting macro alpha from it.

The experiment I’d run first is residualization. Take your existing raw acceleration score and regress/predict it using only conventional information:

A_{i,t}
=
f(\text{NAICS},\,
\Delta Sales,\,
\Delta Capex,\,
\text{PMI},\,
\text{rates},\,
\text{prior }A_{i,t-1},\ldots)
+\epsilon_{i,t}.

Then throw away A temporarily and investigate \epsilon. Aggregate the residuals by industry, examine their cross-sectional dispersion, and test whether they predict future industry orders/production/earnings revisions.

That directly addresses your concern: if the LLM is merely rediscovering cyclicality, residualization should kill it. If something survives, that’s much closer to the unique information you’re looking for.

And I think your granularity gives you another potentially powerful construct. Suppose you calculate:

S_{j,t}=
\frac{\#\{\text{firms in industry }j\text{ newly accelerating}\}}
{\#\{\text{firms in industry }j\}}
-
E[S_{j,t}\mid \text{macro}]

and then map industry j’s capex into the industries that supply its investment goods, rather than into j itself. You effectively get a forward demand signal for suppliers. That seems substantially more economically identified than trying to make one blue aggregate line lead ISM.

So I wouldn’t optimize the weighting until the aggregate chart turns positive. That risks data mining. Use the transcript granularity to construct something conventional macro data fundamentally cannot observe: surprises, transitions, composition, commitment stage, and customer→supplier transmission. If one of those works out of sample, that’s where the defensible alpha is likely to be.

BQNT export scripts to automate dashboard publishing directly within the terminal environment.