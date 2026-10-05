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




Ultimate objective: construct 3–5 economically interpretable transcript-derived leading indicators that can be plotted like my existing blue line against ISM PMI, IP Manufacturing, core capital-goods orders, and private investment.

The indicators must not obtain their predictive relationship mechanically from base effects, inversion, YoY arithmetic, or shifting a coincident cyclical variable.

Start from the firm-level residual/surprise dataset we already built. Do not optimize arbitrary transformations for maximum correlation.

Construct these candidate series:

1. Capex Surprise Breadth
> CSB_t=P(\epsilon_{i,t}>0)-P(\epsilon_{i,t}<0)
>
This measures how broadly management CAPEX intentions are unexpectedly improving.

2. Downside Capex Surprise / Early-Warning Index
Use the p10 or lower-tail residual signal that was already strong in the previous experiment. Test p10/p20/p25 before choosing a definition. Higher values should consistently mean stronger future activity; orient the sign once based on economic interpretation, not based on which sign maximizes correlation.

3. Capex Turning-Point Breadth
At the firm level, identify changes in residual signal from the previous earnings call. Calculate:
> P(\text{negative}\rightarrow\text{positive})
> -
> P(\text{positive}\rightarrow\text{negative})
>
This should capture new corporate turns, rather than the level of the business cycle.

4. Cross-Industry Capex Surprise Breadth
First aggregate residuals within industries. Then calculate the fraction of industries with positive versus negative surprises. Give every industry equal weight initially so that transcript coverage doesn’t determine the result.

5. Capex Surprise Diffusion
Construct a diffusion index centered on 50 from the firm-level residuals, analogous in presentation to a PMI. Positive unexpected CAPEX intentions push it above 50; negative surprises push it below 50.

For each candidate, create a causal real-time series using only information available as of each date. Produce raw, trailing-3m, trailing-6m and trailing-12m versions. No centered smoothing.

Test each against future ISM PMI, ISM New Orders, IP Manufacturing and core capital-goods orders at predetermined horizons 0, 1Q, 2Q, 3Q, 4Q, 6Q and 8Q.

Do not select a different transformation and lead for every target. Look for one or two specifications that have broadly consistent economic timing across targets.

Most importantly, run a horse race against simple cyclical predictors. Test whether the transcript indicator adds forecasting information conditional on current ISM, lagged ISM and the recent change in ISM. If possible also condition on current core-orders growth.

Report:

* full lead/lag correlation curves;
* incremental predictive coefficient;
* incremental out-of-sample R²;
* performance excluding 2020–21;
* performance separately pre/post 2020;
* stability of the chosen lead;
* whether sign and lead were specified ex ante or selected in sample.

Finally, generate presentation charts exactly like my existing chart: Actual Timing on the bottom and Shifted to Align on top. But only make the shifted chart for signals that pass the incremental/out-of-sample tests.

The goal is NOT to maximize correlation. The goal is to find a defensible series where today’s transcript surprises contain information about economic activity that has not yet appeared in conventional cyclical indicators.

Stop searching for the highest-correlation aggregate index. Determine where the transcript residual creates incremental, actionable macro information.

Run four experiments.

1. Forecast horse race. For ISM, ISM New Orders, industrial production, core capital-goods orders, equipment investment and construction spending, estimate a conventional real-time baseline using only lagged target variables and whatever macro variables are already in the dataset. Then add the transcript residual features. Use expanding/rolling pseudo-out-of-sample estimation. Report baseline versus baseline+LLM out-of-sample R², RMSE and forecast errors at 1Q, 2Q, 3Q and 4Q horizons. Do not optimize horizons based on the full sample.

2. Turning-point experiment. Define objective macro turning points and ask whether the residual cross-section improves the probability of identifying a turn over the next 3/6/9 months. Use mean, p10, p25, median, dispersion, negative-surprise breadth, positive-surprise breadth and industry breadth. Evaluate false positives as well as true positives.

3. Regime/conditional information. Determine when residuals matter. Interact them with current ISM level/direction and divide firms by industry. Test whether residual deterioration is particularly informative when ISM is high but weakening, low but improving, etc. Also test whether disagreement/dispersion has information independent of the mean.

4. Sector-to-macro mapping. Build residual indexes by NAICS sector and determine which sectors predict which macro series. Don’t maximize correlation blindly. Establish economically sensible mappings—for example, manufacturing/equipment-related corporate intentions → capital-goods orders/IP; construction-related intentions → construction investment. Then create a macro dashboard from the validated sector signals.

For every result distinguish contemporaneous explanatory power, genuine forecasting power, and turning-point information. Do not call increased in-sample R² alpha. Preserve a locked out-of-sample period and account for multiple testing.

The result that would impress me

Not another chart with two lines that happen to have r=.7.

Imagine presenting this:

Panel 1 — “What companies are saying that the macro data doesn’t know.”

Your Capex Expectations Surprise Index, centered around zero. It doesn’t need to mimic ISM.

Panel 2 — Forecast improvement.

Something as simple as:

\text{Macro-only model}
\quad\rightarrow\quad
\text{Macro + 22,000 firms' transcript surprises}

and then demonstrate that adding the transcript information reduces genuine out-of-sample forecast error for, say, 6-month-ahead manufacturing activity.

That is much harder to dismiss than visual correlation.

Panel 3 — Cross-sectional heat map.

Rows = industries. Columns = quarters/months. Show unexpected acceleration/deceleration. Then say something like:

“The aggregate macro data looks stable, but deterioration is spreading from 3 of 20 industries to 11 of 20.”

That’s a product a macro strategist could actually use.

And there may be an even better use: disagreement

You’ve got something most macro datasets don’t have: thousands of individual corporate observations underneath the aggregate.

Consequently,

\text{Mean residual}

may not be your killer variable. Try:

\text{Dispersion}(\epsilon_{i,t}),
\qquad
P_{10}(\epsilon_{i,t}),
\qquad
P_{90}(\epsilon_{i,t}),

and especially

\text{Breadth}_t=
\%\{\epsilon_i>0\}-\%\{\epsilon_i<0\}.

Your earlier result that the p10 residual appeared particularly informative makes me interested in the lower tail. It could be that macro turns don’t begin with the average company changing its mind. They begin with a minority of companies encountering unexpectedly bad conditions, after which the weakness diffuses.

That’s an intrinsically cross-sectional phenomenon. A conventional macro series can’t give you the same observation.

I’d also broaden beyond CAPEX

Once you’ve proven the residual concept, the really impressive demonstration of the tool isn’t one CAPEX index.

Run the same frozen LLM framework over every transcript and extract independently:

\begin{array}{ccc}
\text{CAPEX} & \text{Hiring} & \text{Demand}\\
\text{Inventories} & \text{Pricing} & \text{Margins}\\
\text{Orders} & \text{Supply chain} & \text{Credit conditions}
\end{array}

Then residualize each one.

Now you potentially have a Corporate Macro Expectations System derived from thousands of management teams.

That’s closer to what central banks already find interesting about earnings-call text: for example, the ECB has used transcript analysis to construct timely corporate sentiment/risk measures, updated during the quarter.  And expectations data have long been shown to contain investment information not subsumed by conventional financial variables. 

Your presentation can then tell a much stronger story:

Traditional data tells us what companies have done. Earnings transcripts tell us what thousands of management teams are planning, where those expectations are changing unexpectedly, and how those changes are diffusing across industries.

Then show one compelling historical episode where the macro indicators still looked benign while your cross-sectional transcript dashboard was changing. Follow that with the full out-of-sample statistics so it’s clear the episode wasn’t cherry-picked.

If I were allocating your research time, I’d spend much less effort now trying to manufacture the perfect blue line from the CAPEX aggregate, and much more on forecast improvement + turning points + cross-sectional diffusion + sector decomposition. That’s where the granularity of your dataset actually gives you something a normal macro strategist doesn’t already have.
      BQNT export scripts to automate dashboard publishing directly within the terminal environment.