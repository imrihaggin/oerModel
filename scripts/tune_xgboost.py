"""Small Optuna-based tuner for XGBoost using time-series CV.

This is an intentionally small run (few trials) for quick experimentation. It saves best params
into artifacts/xgboost_optuna_params.json.
"""
import json
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

FEATURES_PATH = ROOT / "data" / "processed" / "features_processed.csv"
if not FEATURES_PATH.exists():
    raise SystemExit(f"Features not found: {FEATURES_PATH}. Run build-features first.")

df = pd.read_csv(FEATURES_PATH)
# assume first column is date, target named oer_cpi_yoy
if "oer_cpi_yoy" not in df.columns:
    raise SystemExit("Target column 'oer_cpi_yoy' not found in features.")

# Drop date and any NA columns
X = df.drop(columns=[c for c in ["date", "oer_cpi_yoy"] if c in df.columns])
y = df["oer_cpi_yoy"].astype(float)

# simple imputation
X = X.fillna(method="ffill").fillna(0.0)

tscv = TimeSeriesSplit(n_splits=4)


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 5.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "verbosity": 0,
        "objective": "reg:squarederror",
    }

    rmses = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(**params)
        # Some xgboost sklearn wrappers in older/newer builds do not accept
        # early_stopping_rounds as a fit kwarg. Use a plain fit here for
        # compatibility in the quick tuning run (we rely on low n_estimators).
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
    return float(np.mean(rmses))


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8, timeout=60 * 10)
    best = study.best_params
    print("Best params:", best)
    with open(ARTIFACTS / "xgboost_optuna_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Saved best params to artifacts/xgboost_optuna_params.json")
