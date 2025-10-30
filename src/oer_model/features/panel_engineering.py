"""
Panel-Aware Feature Engineering for OER Forecasting

This module implements feature engineering that respects the 6-panel survey
structure of the BLS Housing Survey used to calculate OER.

Key Insight from Research:
The OER index is calculated from a 6-panel rotating survey where each rental
unit is surveyed once every 6 months. Each month, only 1/6 of the sample is
repriced. This creates a mechanical 6-month moving average effect that must
be modeled explicitly.

Panel Structure:
- Panel 1: Surveyed in January, July
- Panel 2: Surveyed in February, August  
- Panel 3: Surveyed in March, September
- Panel 4: Surveyed in April, October
- Panel 5: Surveyed in May, November
- Panel 6: Surveyed in June, December

The monthly OER index is a weighted average of:
- 1/6 current month panel (t)
- 1/6 one-month-old data (t-1)
- 1/6 two-month-old data (t-2)
- ... through t-5

This module creates features that explicitly model this structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def create_panel_weighted_average(
    series: pd.Series,
    n_panels: int = 6,
    alias: Optional[str] = None
) -> pd.Series:
    """
    Create a moving average that mimics the BLS 6-panel survey structure.
    
    The OER methodology means the current month's index reflects:
    - Current panel data (1/6 weight)
    - Previous 5 panels' data (1/6 weight each)
    
    For a market indicator like ZORI, we can create a synthetic "BLS-adjusted"
    version by applying the same panel structure to understand how market
    changes will flow through to official statistics.
    
    Args:
        series: Input time series (e.g., ZORI or market rent index)
        n_panels: Number of panels in the survey (default 6 for OER)
        alias: Optional name for the output series
        
    Returns:
        Panel-weighted moving average series
        
    Example:
        If market rents spike in January, the official OER won't fully
        reflect this until June (when all 6 panels have been repriced).
        This function shows the "BLS view" of market data.
    """
    # Equal weights for each panel
    weights = np.ones(n_panels) / n_panels
    
    # Create rolling window with equal weights
    result = series.rolling(window=n_panels, min_periods=1).apply(
        lambda x: np.average(x, weights=weights[:len(x)])
    )
    
    if alias:
        result.name = alias
    else:
        result.name = f"{series.name}_panel_ma{n_panels}"
        
    return result


def create_panel_velocity_features(
    series: pd.Series,
    n_panels: int = 6
) -> pd.DataFrame:
    """
    Create features capturing the rate of change within panel periods.
    
    These features help models understand momentum and acceleration in
    the underlying market, which predicts how the smoothed official
    statistics will evolve.
    
    Args:
        series: Input time series
        n_panels: Number of panels (default 6)
        
    Returns:
        DataFrame with velocity features
    """
    df = pd.DataFrame(index=series.index)
    base_name = series.name
    
    # Change over one panel period (6 months)
    df[f'{base_name}_panel_change'] = series.pct_change(n_panels) * 100
    
    # Acceleration: change in the rate of change
    df[f'{base_name}_panel_accel'] = df[f'{base_name}_panel_change'].diff()
    
    # Recent vs panel: current value vs panel-averaged value
    panel_ma = create_panel_weighted_average(series, n_panels)
    df[f'{base_name}_vs_panel_avg'] = ((series / panel_ma) - 1) * 100
    
    # This measures how much the "real-time" market has diverged from
    # what the BLS survey structure "sees"
    
    return df


def create_panel_decomposition_features(
    series: pd.Series,
    n_panels: int = 6
) -> pd.DataFrame:
    """
    Decompose a market indicator into components that align with
    the panel survey structure.
    
    This creates features representing:
    1. What each historical panel "sees" 
    2. The contribution of each panel to the current synthetic index
    3. Expected near-term changes as newer panels replace older ones
    
    Args:
        series: Input time series (typically a high-frequency rent index)
        n_panels: Number of panels in survey structure
        
    Returns:
        DataFrame with decomposition features
    """
    df = pd.DataFrame(index=series.index)
    base_name = series.name
    
    # Create lagged versions representing each panel
    for i in range(n_panels):
        df[f'{base_name}_panel_{i+1}'] = series.shift(i)
    
    # Calculate the current "BLS view" (equal-weighted average)
    panel_cols = [f'{base_name}_panel_{i+1}' for i in range(n_panels)]
    df[f'{base_name}_bls_view'] = df[panel_cols].mean(axis=1)
    
    # Calculate what the next month's BLS view will be
    # (drops oldest panel, adds current value)
    next_month_panels = [series] + [series.shift(i) for i in range(1, n_panels)]
    df[f'{base_name}_bls_view_next'] = pd.concat(next_month_panels, axis=1).mean(axis=1)
    
    # Expected change in BLS view (forward-looking)
    df[f'{base_name}_bls_view_delta'] = df[f'{base_name}_bls_view_next'] - df[f'{base_name}_bls_view']
    
    return df


def create_panel_interaction_features(
    market_series: pd.Series,
    oer_series: pd.Series,
    n_panels: int = 6
) -> pd.DataFrame:
    """
    Create features that capture the interaction between real-time market
    indicators and the lagged, smoothed official OER statistic.
    
    These features help models learn regime-dependent relationships, such as:
    - When market rents lead OER by more or less than usual
    - When the relationship breaks down (structural changes)
    - Acceleration/deceleration patterns
    
    Args:
        market_series: High-frequency market indicator (e.g., ZORI)
        oer_series: Official OER statistic
        n_panels: Number of panels
        
    Returns:
        DataFrame with interaction features
    """
    df = pd.DataFrame(index=market_series.index)
    market_name = market_series.name
    oer_name = oer_series.name
    
    # Market YoY vs OER YoY (spread)
    market_yoy = market_series.pct_change(12) * 100
    oer_yoy = oer_series.pct_change(12) * 100
    df[f'{market_name}_oer_spread'] = market_yoy - oer_yoy
    
    # Panel-adjusted market vs OER
    market_panel = create_panel_weighted_average(market_series, n_panels)
    market_panel_yoy = market_panel.pct_change(12) * 100
    df[f'{market_name}_panel_oer_spread'] = market_panel_yoy - oer_yoy
    
    # This measures whether the panel-adjusted market indicator
    # is converging to or diverging from official OER
    
    # Regime indicator: high divergence vs low divergence
    spread_ma = df[f'{market_name}_oer_spread'].rolling(12).mean()
    spread_std = df[f'{market_name}_oer_spread'].rolling(12).std()
    df[f'{market_name}_oer_z_score'] = (df[f'{market_name}_oer_spread'] - spread_ma) / spread_std
    
    # Convergence momentum
    df[f'{market_name}_oer_convergence'] = -df[f'{market_name}_oer_spread'].diff(n_panels)
    # Positive values = spread is narrowing (market converging to OER)
    # Negative values = spread is widening (diverging)
    
    return df


def build_panel_aware_features(
    df: pd.DataFrame,
    market_indicators: list[str],
    oer_column: str = 'oer_cpi',
    n_panels: int = 6
) -> pd.DataFrame:
    """
    Main function to build comprehensive panel-aware features.
    
    This orchestrates all panel-specific transformations and returns
    a complete feature matrix that respects the OER survey structure.
    
    Args:
        df: Input DataFrame with raw series
        market_indicators: List of column names for market indicators (ZORI, ZHVI, etc.)
        oer_column: Column name for OER target variable
        n_panels: Number of panels in survey structure
        
    Returns:
        Enhanced DataFrame with panel-aware features
        
    Example:
        >>> df = pd.DataFrame({
        ...     'oer_cpi': [...],
        ...     'zori': [...],
        ...     'zhvi': [...]
        ... })
        >>> enhanced = build_panel_aware_features(
        ...     df, 
        ...     market_indicators=['zori', 'zhvi'],
        ...     oer_column='oer_cpi'
        ... )
    """
    LOGGER.info("Building panel-aware features (n_panels=%d)", n_panels)
    
    result = df.copy()
    
    for indicator in market_indicators:
        if indicator not in df.columns:
            LOGGER.warning("Market indicator %s not found in DataFrame", indicator)
            continue
            
        LOGGER.info("Processing panel features for %s", indicator)
        
        # 1. Panel-weighted moving average (BLS view)
        result[f'{indicator}_panel_ma'] = create_panel_weighted_average(
            df[indicator], n_panels, alias=f'{indicator}_panel_ma'
        )
        
        # 2. Velocity features
        velocity_df = create_panel_velocity_features(df[indicator], n_panels)
        result = pd.concat([result, velocity_df], axis=1)
        
        # 3. Panel decomposition
        decomp_df = create_panel_decomposition_features(df[indicator], n_panels)
        result = pd.concat([result, decomp_df], axis=1)
        
        # 4. Interaction with OER (if available)
        if oer_column in df.columns:
            interaction_df = create_panel_interaction_features(
                df[indicator], df[oer_column], n_panels
            )
            result = pd.concat([result, interaction_df], axis=1)
    
    LOGGER.info("Panel-aware feature engineering complete. Added %d new features.",
                len(result.columns) - len(df.columns))
    
    return result


def calculate_panel_forecast_adjustment(
    market_forecast: np.ndarray,
    historical_market: pd.Series,
    n_panels: int = 6,
    forecast_horizon: int = 12
) -> np.ndarray:
    """
    Adjust a forecast to account for the panel survey lag structure.
    
    When forecasting OER, a naive model might predict the future value
    directly. However, due to the panel structure, we need to account
    for the fact that the "future" OER will partially reflect "past"
    market conditions (from panels that haven't been updated yet).
    
    This function takes a forecast of market conditions and adjusts it
    to better predict the panel-smoothed OER that will be reported.
    
    Args:
        market_forecast: Array of forecasted market values
        historical_market: Historical market values (used for panel calculation)
        n_panels: Number of panels
        forecast_horizon: Number of periods ahead
        
    Returns:
        Adjusted forecast that accounts for panel lag
        
    Example:
        If forecasting 12 months ahead, the OER value in month 12 will
        actually reflect panels surveyed over months 7-12. This function
        creates the appropriate weighted average.
    """
    adjusted_forecast = np.zeros(forecast_horizon)
    
    for h in range(forecast_horizon):
        # At horizon h, the OER reflects:
        # - Forecast periods 0 to h (weight = min(h+1, n_panels) / n_panels)
        # - Historical periods (weight = max(0, n_panels - h - 1) / n_panels)
        
        if h < n_panels - 1:
            # Still using some historical data
            n_forecast_panels = h + 1
            n_historical_panels = n_panels - n_forecast_panels
            
            forecast_component = market_forecast[:n_forecast_panels].mean()
            historical_component = historical_market.iloc[-(n_historical_panels):].mean()
            
            adjusted_forecast[h] = (
                forecast_component * n_forecast_panels / n_panels +
                historical_component * n_historical_panels / n_panels
            )
        else:
            # Fully based on forecast periods
            adjusted_forecast[h] = market_forecast[max(0, h-n_panels+1):h+1].mean()
    
    return adjusted_forecast
