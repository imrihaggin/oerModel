"""
Forecast combination and ensemble methods.

This module provides multiple ensemble strategies for combining
forecasts from different models, including:
- Simple averaging (equal weights)
- Performance-weighted averaging (inverse error weighting)
- Bayesian Model Averaging (BMA)
- Variance-weighted ensembles
- Stacking meta-models
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def equal_weight_average(forecasts: Union[dict[str, pd.Series], pd.DataFrame]) -> pd.Series:
    """
    Return an equal-weight ensemble of forecasts.
    
    Args:
        forecasts: Either dict mapping model names to forecast Series,
                  or DataFrame where columns are model forecasts
                  
    Returns:
        Series of averaged forecasts
    """
    if isinstance(forecasts, dict):
        if not forecasts:
            raise ValueError("No forecasts provided")
        aligned = pd.concat(forecasts.values(), axis=1)
        aligned.columns = list(forecasts.keys())
        return aligned.mean(axis=1)
    else:
        return forecasts.mean(axis=1)


def performance_weighted_average(
    forecasts: pd.DataFrame,
    errors: pd.DataFrame,
    method: str = "inverse_rmse"
) -> pd.Series:
    """
    Combine forecasts using performance-based weights.
    
    Better-performing models receive higher weights based on their
    historical forecast accuracy.
    
    Args:
        forecasts: DataFrame where each column is a model's forecast
        errors: DataFrame where each column is a model's historical errors
        method: Weighting method - 'inverse_rmse', 'inverse_mae', 'squared_inverse'
        
    Returns:
        Series of weighted forecasts
        
    Example:
        >>> forecasts = pd.DataFrame({'model_a': [1.2, 1.3], 'model_b': [1.1, 1.2]})
        >>> errors = pd.DataFrame({'model_a': [0.1, 0.15], 'model_b': [0.05, 0.08]})
        >>> ensemble = performance_weighted_average(forecasts, errors)
    """
    LOGGER.info("Computing performance-weighted ensemble (%s)", method)
    
    # Calculate weights from errors
    if method == "inverse_rmse":
        # Weight inversely proportional to RMSE
        rmse = np.sqrt((errors ** 2).mean())
        weights = 1.0 / rmse
    elif method == "inverse_mae":
        # Weight inversely proportional to MAE
        mae = errors.abs().mean()
        weights = 1.0 / mae
    elif method == "squared_inverse":
        # Weight inversely proportional to squared RMSE (penalizes poor models more)
        rmse = np.sqrt((errors ** 2).mean())
        weights = 1.0 / (rmse ** 2)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    # Log weights
    for model, weight in weights.items():
        LOGGER.info("  %s: %.4f", model, weight)
    
    # Compute weighted average
    weighted_forecast = (forecasts * weights).sum(axis=1)
    
    return weighted_forecast


def bayesian_model_averaging(
    forecasts: pd.DataFrame,
    errors: pd.DataFrame,
    prior: Optional[Dict[str, float]] = None,
    temperature: float = 1.0
) -> pd.Series:
    """
    Bayesian Model Averaging (BMA) for forecast combination.
    
    Weights models by their posterior probability given the data,
    computed using their likelihood (based on forecast errors).
    
    Args:
        forecasts: DataFrame where each column is a model's forecast
        errors: DataFrame where each column is a model's historical errors
        prior: Optional dict of prior probabilities for each model (uniform if None)
        temperature: Temperature parameter for softmax (lower = more concentrated)
        
    Returns:
        Series of BMA-weighted forecasts
        
    Theory:
        P(model|data) ∝ P(data|model) * P(model)
        
        Where:
        - P(model) is the prior (uniform if not specified)
        - P(data|model) is the likelihood (based on forecast accuracy)
        - We use squared error to approximate log-likelihood
    """
    LOGGER.info("Computing Bayesian Model Averaging ensemble")
    
    # Uniform prior if not specified
    if prior is None:
        prior = {col: 1.0 / len(forecasts.columns) for col in forecasts.columns}
    
    # Convert prior to array
    prior_array = np.array([prior.get(col, 1.0 / len(forecasts.columns)) 
                           for col in forecasts.columns])
    
    # Compute log-likelihood from errors (using negative squared error)
    # Better models (lower errors) have higher likelihood
    mse = (errors ** 2).mean()
    log_likelihood = -mse / temperature
    
    # Compute posterior (unnormalized)
    log_posterior = log_likelihood + np.log(prior_array)
    
    # Apply softmax to get normalized weights
    max_log_post = log_posterior.max()
    weights = np.exp(log_posterior - max_log_post)
    weights = weights / weights.sum()
    
    # Convert to Series for easier manipulation
    weights_series = pd.Series(weights, index=forecasts.columns)
    
    # Log weights
    LOGGER.info("BMA posterior weights:")
    for model, weight in weights_series.items():
        LOGGER.info("  %s: %.4f", model, weight)
    
    # Compute weighted average
    bma_forecast = (forecasts * weights_series).sum(axis=1)
    
    return bma_forecast


def variance_weighted_ensemble(
    forecasts: pd.DataFrame,
    prediction_intervals: Optional[pd.DataFrame] = None,
    rolling_std: Optional[pd.DataFrame] = None,
    window: int = 12
) -> pd.Series:
    """
    Weight forecasts by their prediction uncertainty.
    
    Models with higher uncertainty (wider prediction intervals or
    higher forecast variance) receive lower weights.
    
    Args:
        forecasts: DataFrame where each column is a model's forecast
        prediction_intervals: Optional DataFrame of prediction interval widths
        rolling_std: Optional DataFrame of rolling forecast standard deviations
        window: Window size for computing rolling std if not provided
        
    Returns:
        Series of variance-weighted forecasts
    """
    LOGGER.info("Computing variance-weighted ensemble")
    
    # Determine uncertainty measure
    if prediction_intervals is not None:
        # Use prediction interval widths
        uncertainty = prediction_intervals.mean()
    elif rolling_std is not None:
        # Use provided rolling std
        uncertainty = rolling_std.mean()
    else:
        # Compute rolling std from forecasts
        uncertainty = forecasts.rolling(window=window).std().mean()
    
    # Weight inversely proportional to uncertainty
    weights = 1.0 / (uncertainty + 1e-6)  # Add small constant to avoid division by zero
    weights = weights / weights.sum()
    
    # Log weights
    LOGGER.info("Variance-based weights:")
    for model, weight in weights.items():
        LOGGER.info("  %s: %.4f", model, weight)
    
    # Compute weighted average
    weighted_forecast = (forecasts * weights).sum(axis=1)
    
    return weighted_forecast


def stacking_ensemble(
    train_forecasts: pd.DataFrame,
    train_actuals: pd.Series,
    test_forecasts: pd.DataFrame,
    meta_model: str = "ridge",
    **meta_model_kwargs
) -> pd.Series:
    """
    Stacking ensemble using a meta-model to learn optimal weights.
    
    Trains a secondary model (meta-model) to combine base model forecasts
    optimally. This can capture nonlinear relationships and interactions.
    
    Args:
        train_forecasts: Training forecasts from base models (n_samples x n_models)
        train_actuals: Training actual values
        test_forecasts: Test forecasts from base models (n_samples x n_models)
        meta_model: Type of meta-model ('ridge', 'lasso', 'rf', 'custom')
        **meta_model_kwargs: Additional arguments for meta-model
        
    Returns:
        Series of stacked forecasts for test set
        
    Example:
        >>> train_fc = pd.DataFrame({'xgb': [1.1, 1.2], 'tft': [1.0, 1.3]})
        >>> train_y = pd.Series([1.05, 1.25])
        >>> test_fc = pd.DataFrame({'xgb': [1.3], 'tft': [1.2]})
        >>> stacked = stacking_ensemble(train_fc, train_y, test_fc, meta_model='ridge')
    """
    LOGGER.info("Training stacking ensemble with %s meta-model", meta_model)
    
    # Initialize meta-model
    if meta_model == "ridge":
        meta = Ridge(alpha=meta_model_kwargs.get('alpha', 1.0), positive=True)
    elif meta_model == "lasso":
        meta = Lasso(alpha=meta_model_kwargs.get('alpha', 0.01), positive=True)
    elif meta_model == "rf":
        meta = RandomForestRegressor(
            n_estimators=meta_model_kwargs.get('n_estimators', 100),
            max_depth=meta_model_kwargs.get('max_depth', 3),
            random_state=42
        )
    elif meta_model == "custom":
        meta = meta_model_kwargs.get('model')
        if meta is None:
            raise ValueError("Must provide 'model' kwarg when meta_model='custom'")
    else:
        raise ValueError(f"Unknown meta-model type: {meta_model}")
    
    # Train meta-model
    meta.fit(train_forecasts, train_actuals)
    
    # Log learned weights (if linear model)
    if hasattr(meta, 'coef_'):
        weights = pd.Series(meta.coef_, index=train_forecasts.columns)
        weights = weights / weights.sum()  # Normalize
        LOGGER.info("Learned stacking weights:")
        for model, weight in weights.items():
            LOGGER.info("  %s: %.4f", model, weight)
    
    # Generate stacked predictions
    stacked_forecast = pd.Series(
        meta.predict(test_forecasts),
        index=test_forecasts.index
    )
    
    return stacked_forecast


def dynamic_weighted_ensemble(
    forecasts: pd.DataFrame,
    errors: pd.DataFrame,
    window: int = 12,
    method: str = "inverse_rmse"
) -> pd.Series:
    """
    Dynamic ensemble with time-varying weights.
    
    Recomputes weights at each time step based on recent performance,
    allowing the ensemble to adapt to changing model performance.
    
    Args:
        forecasts: DataFrame where each column is a model's forecast
        errors: DataFrame where each column is a model's historical errors
        window: Window size for computing recent performance
        method: Weighting method (same as performance_weighted_average)
        
    Returns:
        Series of dynamically weighted forecasts
    """
    LOGGER.info("Computing dynamic weighted ensemble (window=%d)", window)
    
    ensemble_forecast = pd.Series(index=forecasts.index, dtype=float)
    
    for idx in forecasts.index:
        # Get recent errors up to current point
        recent_errors = errors.loc[:idx].tail(window)
        
        if len(recent_errors) < 2:
            # Not enough history, use equal weights
            ensemble_forecast.loc[idx] = forecasts.loc[idx].mean()
            continue
        
        # Compute weights from recent errors
        if method == "inverse_rmse":
            rmse = np.sqrt((recent_errors ** 2).mean())
            weights = 1.0 / rmse
        elif method == "inverse_mae":
            mae = recent_errors.abs().mean()
            weights = 1.0 / mae
        else:
            rmse = np.sqrt((recent_errors ** 2).mean())
            weights = 1.0 / (rmse ** 2)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute weighted forecast for this time step
        ensemble_forecast.loc[idx] = (forecasts.loc[idx] * weights).sum()
    
    return ensemble_forecast


def robust_ensemble(
    forecasts: pd.DataFrame,
    method: str = "median",
    trim_pct: float = 0.1
) -> pd.Series:
    """
    Robust ensemble that is less sensitive to outlier forecasts.
    
    Args:
        forecasts: DataFrame where each column is a model's forecast
        method: Robust method - 'median', 'trimmed_mean', 'winsorized_mean'
        trim_pct: Percentage to trim from each tail (for trimmed/winsorized)
        
    Returns:
        Series of robustly combined forecasts
    """
    LOGGER.info("Computing robust ensemble (%s)", method)
    
    if method == "median":
        # Simple median - most robust
        ensemble = forecasts.median(axis=1)
    elif method == "trimmed_mean":
        # Remove extreme values from each tail
        def trimmed_mean(x):
            sorted_x = np.sort(x.dropna())
            n_trim = int(len(sorted_x) * trim_pct)
            if n_trim > 0:
                return sorted_x[n_trim:-n_trim].mean()
            return sorted_x.mean()
        ensemble = forecasts.apply(trimmed_mean, axis=1)
    elif method == "winsorized_mean":
        # Replace extreme values with percentile values
        def winsorized_mean(x):
            sorted_x = np.sort(x.dropna())
            lower = np.percentile(sorted_x, trim_pct * 100)
            upper = np.percentile(sorted_x, 100 - trim_pct * 100)
            clipped = np.clip(sorted_x, lower, upper)
            return clipped.mean()
        ensemble = forecasts.apply(winsorized_mean, axis=1)
    else:
        raise ValueError(f"Unknown robust method: {method}")
    
    return ensemble


def optimal_ensemble(
    train_forecasts: pd.DataFrame,
    train_actuals: pd.Series,
    test_forecasts: pd.DataFrame,
    method: str = "auto",
    cv_folds: int = 5
) -> pd.Series:
    """
    Automatically select and apply the best ensemble method.
    
    Evaluates multiple ensemble methods using cross-validation on
    the training set and applies the best-performing method to test.
    
    Args:
        train_forecasts: Training forecasts from base models
        train_actuals: Training actual values
        test_forecasts: Test forecasts from base models
        method: 'auto' (select best) or specific method name
        cv_folds: Number of CV folds for method selection
        
    Returns:
        Series of optimally combined forecasts
    """
    LOGGER.info("Selecting optimal ensemble method")
    
    if method != "auto":
        # User specified method
        if method == "equal":
            return equal_weight_average(test_forecasts)
        elif method == "stacking":
            return stacking_ensemble(train_forecasts, train_actuals, test_forecasts)
        # Add other methods as needed
    
    # Evaluate candidate methods using CV
    methods = {
        'equal': lambda: equal_weight_average(train_forecasts),
        'median': lambda: robust_ensemble(train_forecasts, method='median'),
        'stacking_ridge': lambda: None,  # Placeholder for stacking
        'stacking_rf': lambda: None
    }
    
    best_method = None
    best_score = float('inf')
    
    # Simple holdout evaluation for each method
    split_idx = int(len(train_forecasts) * 0.8)
    cv_train_fc = train_forecasts.iloc[:split_idx]
    cv_test_fc = train_forecasts.iloc[split_idx:]
    cv_train_y = train_actuals.iloc[:split_idx]
    cv_test_y = train_actuals.iloc[split_idx:]
    
    # Equal weight
    pred = equal_weight_average(cv_test_fc)
    score = np.sqrt(((pred - cv_test_y) ** 2).mean())
    LOGGER.info("  equal: RMSE=%.4f", score)
    if score < best_score:
        best_score = score
        best_method = 'equal'
    
    # Median
    pred = robust_ensemble(cv_test_fc, method='median')
    score = np.sqrt(((pred - cv_test_y) ** 2).mean())
    LOGGER.info("  median: RMSE=%.4f", score)
    if score < best_score:
        best_score = score
        best_method = 'median'
    
    # Stacking Ridge
    pred = stacking_ensemble(cv_train_fc, cv_train_y, cv_test_fc, meta_model='ridge')
    score = np.sqrt(((pred - cv_test_y) ** 2).mean())
    LOGGER.info("  stacking_ridge: RMSE=%.4f", score)
    if score < best_score:
        best_score = score
        best_method = 'stacking_ridge'
    
    LOGGER.info("Selected best method: %s (CV RMSE=%.4f)", best_method, best_score)
    
    # Apply best method to full train/test
    if best_method == 'equal':
        return equal_weight_average(test_forecasts)
    elif best_method == 'median':
        return robust_ensemble(test_forecasts, method='median')
    elif best_method == 'stacking_ridge':
        return stacking_ensemble(train_forecasts, train_actuals, test_forecasts, meta_model='ridge')
    
    # Fallback
    return equal_weight_average(test_forecasts)
