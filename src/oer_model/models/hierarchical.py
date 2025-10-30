"""
Hierarchical CPI Forecasting Models

This module implements bottom-up and hierarchical forecasting approaches
that leverage the known structure of the CPI index.

CPI Hierarchy:
    CPI (Headline)
    └── Shelter
        ├── Rent of Primary Residence
        └── Owners' Equivalent Rent (OER)  ← Our target

Research motivation:
- Modeling components separately and aggregating can improve accuracy
- Higher-level aggregates provide stability and context to volatile sub-components
- Hierarchical models can enforce consistency across the CPI structure
- Deep learning architectures (HRNN) can explicitly leverage this structure

References:
[22, 23] Component-wise inflation forecasting
[24, 25] Hierarchical RNN for CPI forecasting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .base import ForecastModel
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CPIHierarchy:
    """
    Defines the structure and weights of CPI components.
    
    Weights should sum to 1.0 within each level and represent the
    relative importance as defined by BLS expenditure weights.
    """
    # Shelter component splits (approximate, update with current BLS weights)
    shelter_rent_weight: float = 0.32  # Rent of primary residence
    shelter_oer_weight: float = 0.68   # OER (larger because more homeowners)
    
    # CPI-level (if modeling headline)
    shelter_in_cpi_weight: float = 0.33  # Shelter is ~33% of CPI
    
    def validate(self):
        """Ensure weights are properly normalized."""
        shelter_total = self.shelter_rent_weight + self.shelter_oer_weight
        if not np.isclose(shelter_total, 1.0, atol=0.01):
            raise ValueError(f"Shelter weights must sum to 1.0, got {shelter_total}")


class BottomUpEnsembleModel(ForecastModel):
    """
    Bottom-up hierarchical forecasting model.
    
    Strategy:
    1. Forecast OER and Rent separately using component-specific models
    2. Aggregate to Shelter using BLS expenditure weights
    3. Optionally combine with other CPI components for headline forecast
    
    This approach allows each component to use its most relevant predictors
    while maintaining consistency with the CPI hierarchy.
    """
    
    def __init__(
        self,
        name: str = "bottom_up",
        oer_model: Optional[ForecastModel] = None,
        rent_model: Optional[ForecastModel] = None,
        hierarchy: Optional[CPIHierarchy] = None
    ):
        super().__init__(name)
        self.hierarchy = hierarchy or CPIHierarchy()
        self.hierarchy.validate()
        
        # Component models (can be any ForecastModel subclass)
        self.oer_model = oer_model
        self.rent_model = rent_model
        self.fitted = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_rent: Optional[pd.DataFrame] = None,
        y_rent: Optional[pd.Series] = None
    ) -> "BottomUpEnsembleModel":
        """
        Fit component models.
        
        Args:
            X: Features for OER
            y: OER target
            X_rent: Features for Rent (optional, for full hierarchy)
            y_rent: Rent target (optional)
        """
        LOGGER.info("Fitting bottom-up hierarchical model")
        
        # Fit OER model
        if self.oer_model is None:
            # Default to Ridge regression
            from sklearn.linear_model import Ridge
            self.oer_model = Ridge(alpha=1.0)
        
        LOGGER.info("Fitting OER component model")
        if hasattr(self.oer_model, 'fit'):
            self.oer_model.fit(X, y)
        
        # Fit Rent model if data provided
        if X_rent is not None and y_rent is not None and self.rent_model is not None:
            LOGGER.info("Fitting Rent component model")
            if hasattr(self.rent_model, 'fit'):
                self.rent_model.fit(X_rent, y_rent)
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, X_rent: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate bottom-up predictions.
        
        If only OER features provided, returns OER forecast.
        If both OER and Rent features provided, returns aggregated Shelter forecast.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Predict OER
        oer_pred = self.oer_model.predict(X)
        
        # If no rent data, return OER only
        if X_rent is None or self.rent_model is None:
            return oer_pred
        
        # Predict Rent
        rent_pred = self.rent_model.predict(X_rent)
        
        # Aggregate to Shelter using BLS weights
        shelter_pred = (
            self.hierarchy.shelter_oer_weight * oer_pred +
            self.hierarchy.shelter_rent_weight * rent_pred
        )
        
        LOGGER.info("Bottom-up aggregation: OER (%.1f%%) + Rent (%.1f%%) -> Shelter",
                   self.hierarchy.shelter_oer_weight * 100,
                   self.hierarchy.shelter_rent_weight * 100)
        
        return shelter_pred


class ReconciliationModel(ForecastModel):
    """
    Hierarchical forecast reconciliation using optimal combination.
    
    This approach:
    1. Generates "base" forecasts at all levels (OER, Rent, Shelter)
    2. Uses optimization to reconcile forecasts so they are coherent
       (i.e., components aggregate to the total)
    3. Typically reduces forecast error compared to unreconciled forecasts
    
    Based on optimal reconciliation methods from Hyndman et al.
    """
    
    def __init__(
        self,
        name: str = "reconciled",
        base_models: Optional[Dict[str, ForecastModel]] = None,
        hierarchy: Optional[CPIHierarchy] = None,
        method: str = "ols"  # 'ols', 'wls', or 'mint'
    ):
        super().__init__(name)
        self.hierarchy = hierarchy or CPIHierarchy()
        self.hierarchy.validate()
        self.base_models = base_models or {}
        self.method = method
        self.reconciliation_matrix = None
        
    def _create_summation_matrix(self) -> np.ndarray:
        """
        Create the summation matrix S that defines hierarchical structure.
        
        For our simple 2-level hierarchy:
        Shelter = w1*OER + w2*Rent
        
        Returns:
            S matrix of shape (n_bottom, n_total) where:
            - n_bottom = number of bottom-level series (2: OER, Rent)
            - n_total = n_bottom + n_top (3: OER, Rent, Shelter)
        """
        # Summing matrix for our hierarchy
        # [OER, Rent] -> [Shelter, OER, Rent]
        S = np.array([
            [self.hierarchy.shelter_oer_weight, self.hierarchy.shelter_rent_weight],  # Shelter
            [1.0, 0.0],  # OER
            [0.0, 1.0]   # Rent
        ])
        return S
    
    def _compute_reconciliation_matrix(
        self,
        S: np.ndarray,
        forecast_errors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the reconciliation matrix G for optimal forecast combination.
        
        Args:
            S: Summation matrix
            forecast_errors: Historical forecast errors for variance estimation
            
        Returns:
            G matrix used to reconcile base forecasts: y_reconciled = S @ G @ y_base
        """
        if self.method == "ols":
            # OLS reconciliation (simplest, equal weights)
            # G = (S'S)^{-1}S'
            G = np.linalg.inv(S.T @ S) @ S.T
            
        elif self.method == "wls":
            # WLS reconciliation (weights by variance of base forecasts)
            if forecast_errors is None:
                LOGGER.warning("WLS requires forecast_errors, falling back to OLS")
                G = np.linalg.inv(S.T @ S) @ S.T
            else:
                # Estimate variances from errors
                W = np.diag(1 / np.var(forecast_errors, axis=0))
                G = np.linalg.inv(S.T @ W @ S) @ S.T @ W
                
        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")
        
        return G
    
    def fit(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y_dict: Dict[str, pd.Series]
    ) -> "ReconciliationModel":
        """
        Fit base models for each level of the hierarchy.
        
        Args:
            X_dict: Dictionary mapping level names to feature matrices
                   e.g., {'oer': X_oer, 'rent': X_rent, 'shelter': X_shelter}
            y_dict: Dictionary mapping level names to targets
        """
        LOGGER.info("Fitting reconciliation model with %d hierarchy levels", len(y_dict))
        
        # Fit base models
        for level, model in self.base_models.items():
            if level in X_dict and level in y_dict:
                LOGGER.info("Fitting base model for %s", level)
                model.fit(X_dict[level], y_dict[level])
        
        # Compute reconciliation matrix
        S = self._create_summation_matrix()
        self.reconciliation_matrix = self._compute_reconciliation_matrix(S)
        
        LOGGER.info("Reconciliation matrix computed (method=%s)", self.method)
        return self
    
    def predict(
        self,
        X_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Generate reconciled forecasts.
        
        Args:
            X_dict: Dictionary of features for each level
            
        Returns:
            Dictionary of reconciled predictions for each level
        """
        # Generate base forecasts
        base_forecasts = {}
        for level, model in self.base_models.items():
            if level in X_dict:
                base_forecasts[level] = model.predict(X_dict[level])
        
        # Stack base forecasts
        # Order: [Shelter, OER, Rent]
        y_base = np.column_stack([
            base_forecasts.get('shelter', np.zeros(len(next(iter(base_forecasts.values()))))),
            base_forecasts.get('oer', np.zeros(len(next(iter(base_forecasts.values()))))),
            base_forecasts.get('rent', np.zeros(len(next(iter(base_forecasts.values())))))
        ])
        
        # Reconcile
        S = self._create_summation_matrix()
        y_reconciled = (S @ self.reconciliation_matrix @ y_base.T).T
        
        # Unpack reconciled forecasts
        reconciled = {
            'shelter': y_reconciled[:, 0],
            'oer': y_reconciled[:, 1],
            'rent': y_reconciled[:, 2]
        }
        
        LOGGER.info("Forecasts reconciled. Max adjustment: %.4f",
                   np.max(np.abs(y_reconciled - y_base)))
        
        return reconciled


class HierarchicalFeatureModel(ForecastModel):
    """
    Single model that uses hierarchical features as inputs.
    
    Instead of separate models, this creates features representing
    higher-level aggregates and uses them to improve component forecasts.
    
    Features include:
    - Lagged Shelter inflation (provides top-down context)
    - Spread between OER and Rent (mean-reversion signal)
    - Historical relationship between levels
    """
    
    def __init__(
        self,
        name: str = "hierarchical_features",
        base_model: Optional[Any] = None,
        hierarchy: Optional[CPIHierarchy] = None
    ):
        super().__init__(name)
        self.hierarchy = hierarchy or CPIHierarchy()
        self.base_model = base_model
        
    def _engineer_hierarchical_features(
        self,
        X: pd.DataFrame,
        shelter_series: Optional[pd.Series] = None,
        rent_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Create features that leverage the CPI hierarchy.
        
        Args:
            X: Base feature matrix
            shelter_series: Shelter CPI time series
            rent_series: Rent CPI time series
            
        Returns:
            Enhanced feature matrix
        """
        X_enhanced = X.copy()
        
        if shelter_series is not None:
            # Top-down features: Shelter provides context
            X_enhanced['shelter_yoy'] = shelter_series.pct_change(12) * 100
            X_enhanced['shelter_yoy_lag3'] = X_enhanced['shelter_yoy'].shift(3)
            X_enhanced['shelter_yoy_lag6'] = X_enhanced['shelter_yoy'].shift(6)
            X_enhanced['shelter_mom'] = shelter_series.pct_change(1) * 100
            
        if rent_series is not None and 'oer_yoy' in X.columns:
            # Cross-component features: OER vs Rent relationship
            rent_yoy = rent_series.pct_change(12) * 100
            X_enhanced['oer_rent_spread'] = X['oer_yoy'] - rent_yoy
            X_enhanced['oer_rent_spread_ma6'] = X_enhanced['oer_rent_spread'].rolling(6).mean()
            
            # Mean reversion feature
            spread_mean = X_enhanced['oer_rent_spread'].rolling(24).mean()
            X_enhanced['oer_rent_spread_zscore'] = (
                (X_enhanced['oer_rent_spread'] - spread_mean) / 
                X_enhanced['oer_rent_spread'].rolling(24).std()
            )
        
        return X_enhanced
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        shelter_series: Optional[pd.Series] = None,
        rent_series: Optional[pd.Series] = None
    ) -> "HierarchicalFeatureModel":
        """Fit model with hierarchical features."""
        X_enhanced = self._engineer_hierarchical_features(X, shelter_series, rent_series)
        
        if self.base_model is None:
            self.base_model = Ridge(alpha=1.0)
        
        # Drop NaN from feature engineering
        valid_idx = ~X_enhanced.isnull().any(axis=1)
        self.base_model.fit(X_enhanced[valid_idx], y[valid_idx])
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        shelter_series: Optional[pd.Series] = None,
        rent_series: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Generate predictions using hierarchical features."""
        X_enhanced = self._engineer_hierarchical_features(X, shelter_series, rent_series)
        return self.base_model.predict(X_enhanced)


def build_hierarchical_dataset(
    data: pd.DataFrame,
    oer_col: str = 'oer_cpi',
    rent_col: str = 'rent_cpi',
    shelter_col: Optional[str] = None,
    hierarchy: Optional[CPIHierarchy] = None
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for hierarchical modeling by creating separate datasets
    for each level with appropriate features.
    
    Args:
        data: Raw data with all CPI components
        oer_col: Column name for OER
        rent_col: Column name for Rent
        shelter_col: Column name for Shelter (if available)
        hierarchy: CPI hierarchy specification
        
    Returns:
        Dictionary with 'oer', 'rent', 'shelter' keys mapping to feature/target pairs
    """
    hierarchy = hierarchy or CPIHierarchy()
    result = {}
    
    # OER dataset
    if oer_col in data.columns:
        result['oer'] = {
            'y': data[oer_col].pct_change(12) * 100,
            'X': data.drop(columns=[oer_col, rent_col] if rent_col in data.columns else [oer_col])
        }
    
    # Rent dataset
    if rent_col in data.columns:
        result['rent'] = {
            'y': data[rent_col].pct_change(12) * 100,
            'X': data.drop(columns=[oer_col, rent_col] if oer_col in data.columns else [rent_col])
        }
    
    # Shelter dataset (either actual or constructed)
    if shelter_col in data.columns:
        result['shelter'] = {
            'y': data[shelter_col].pct_change(12) * 100,
            'X': data.drop(columns=[shelter_col])
        }
    elif oer_col in data.columns and rent_col in data.columns:
        # Construct shelter from components
        shelter_synthetic = (
            hierarchy.shelter_oer_weight * data[oer_col] +
            hierarchy.shelter_rent_weight * data[rent_col]
        )
        result['shelter'] = {
            'y': shelter_synthetic.pct_change(12) * 100,
            'X': data.drop(columns=[oer_col, rent_col])
        }
    
    return result
