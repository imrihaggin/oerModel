"""
Reporting helpers for evaluation outputs with interpretability integration.

This module provides comprehensive reporting functionality including:
- Standard backtest aggregation
- SHAP value computation for tree-based models
- TFT attention heatmap generation
- Interpretation report assembly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Optional dependencies for interpretability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    LOGGER.warning("SHAP not available. Install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    LOGGER.warning("Matplotlib/Seaborn not available for plotting")


def aggregate_backtest_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling backtest metrics by model."""
    metrics = [col for col in ["rmse", "mae", "mape"] if col in results.columns]
    return results.groupby("model")[metrics].mean().sort_values("rmse")


def prepare_dashboard_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Reshape backtest results for plotting."""
    pivot = results.pivot_table(index="timestamp", columns="model", values="y_pred")
    pivot["actual"] = results.drop_duplicates("timestamp").set_index("timestamp")["y_true"]
    return pivot.sort_index()


# ============================================================================
# SHAP Integration for Tree-Based Models
# ============================================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_samples: int = 500
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Fitted model (must have predict method)
        X: Feature matrix
        model_type: Type of model ('tree', 'linear', 'kernel')
        max_samples: Maximum samples to use (for computational efficiency)
        
    Returns:
        Dictionary containing:
        - shap_values: SHAP value matrix
        - base_value: Expected value
        - feature_names: Feature names
        - data: Sample data used
    """
    if not HAS_SHAP:
        LOGGER.warning("SHAP not installed, skipping SHAP computation")
        return None
    
    LOGGER.info("Computing SHAP values for interpretability")
    
    try:
        # Subsample if needed
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X
        
        # Create appropriate explainer
        if model_type == "tree":
            # For XGBoost, LightGBM, CatBoost, RandomForest
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            # For linear models
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # Kernel explainer (model-agnostic, slower)
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        result = {
            'shap_values': shap_values,
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None,
            'feature_names': X_sample.columns.tolist(),
            'data': X_sample,
            'explainer': explainer
        }
        
        LOGGER.info("SHAP computation complete for %d samples", len(X_sample))
        return result
        
    except Exception as e:
        LOGGER.error("Failed to compute SHAP values: %s", e)
        return None


def plot_shap_summary(
    shap_result: Dict[str, Any],
    output_path: Optional[Path] = None,
    max_display: int = 15,
    plot_type: str = "dot"
) -> Optional[plt.Figure]:
    """
    Create SHAP summary plot showing feature importance.
    
    Args:
        shap_result: Output from compute_shap_values
        output_path: Path to save plot
        max_display: Number of features to display
        plot_type: 'dot' (beeswarm), 'bar', or 'violin'
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_SHAP or not HAS_PLOTTING:
        LOGGER.warning("Required libraries not available for SHAP plotting")
        return None
    
    if shap_result is None:
        LOGGER.warning("No SHAP result provided")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == "dot":
            shap.summary_plot(
                shap_result['shap_values'],
                shap_result['data'],
                max_display=max_display,
                show=False
            )
        elif plot_type == "bar":
            shap.summary_plot(
                shap_result['shap_values'],
                shap_result['data'],
                plot_type="bar",
                max_display=max_display,
                show=False
            )
        elif plot_type == "violin":
            shap.summary_plot(
                shap_result['shap_values'],
                shap_result['data'],
                plot_type="violin",
                max_display=max_display,
                show=False
            )
        
        plt.title('Feature Importance (SHAP Values)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            LOGGER.info("SHAP summary plot saved to %s", output_path)
        
        return fig
        
    except Exception as e:
        LOGGER.error("Failed to create SHAP plot: %s", e)
        return None


def plot_shap_waterfall(
    shap_result: Dict[str, Any],
    sample_idx: int = -1,
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Shows how each feature contributes to pushing the prediction
    from the base value to the final value.
    
    Args:
        shap_result: Output from compute_shap_values
        sample_idx: Index of sample to explain (-1 for last)
        output_path: Path to save plot
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_SHAP or not HAS_PLOTTING:
        return None
    
    if shap_result is None:
        return None
    
    try:
        # Create explanation object for the sample
        explainer = shap_result['explainer']
        data = shap_result['data']
        
        # Get SHAP values for specific sample
        shap_values_sample = explainer(data.iloc[[sample_idx]])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values_sample[0], show=False)
        plt.title(f'Feature Contributions for Sample {sample_idx}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            LOGGER.info("SHAP waterfall plot saved to %s", output_path)
        
        return fig
        
    except Exception as e:
        LOGGER.error("Failed to create SHAP waterfall plot: %s", e)
        return None


def plot_shap_dependence(
    shap_result: Dict[str, Any],
    feature_name: str,
    interaction_feature: Optional[str] = None,
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create SHAP dependence plot for a specific feature.
    
    Shows how feature values affect predictions (with optional interaction).
    
    Args:
        shap_result: Output from compute_shap_values
        feature_name: Feature to plot
        interaction_feature: Optional feature to color by
        output_path: Path to save plot
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_SHAP or not HAS_PLOTTING:
        return None
    
    if shap_result is None:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_name,
            shap_result['shap_values'],
            shap_result['data'],
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence: {feature_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            LOGGER.info("SHAP dependence plot saved to %s", output_path)
        
        return fig
        
    except Exception as e:
        LOGGER.error("Failed to create SHAP dependence plot: %s", e)
        return None


# ============================================================================
# TFT Attention Integration
# ============================================================================

def extract_tft_attention(
    tft_model: Any,
    X: pd.DataFrame,
    sample_idx: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract attention patterns from a fitted TFT model.
    
    Args:
        tft_model: Fitted TFT model (InterpretableTFT instance)
        X: Feature matrix
        sample_idx: Specific sample to analyze (None = last)
        
    Returns:
        Dictionary containing:
        - attention: Attention weight matrix
        - time_labels: Labels for time steps
        - aggregated: Whether attention is aggregated across heads
    """
    LOGGER.info("Extracting attention patterns from TFT model")
    
    try:
        # Check if model has attention extraction method
        if not hasattr(tft_model, 'get_attention_patterns'):
            LOGGER.warning("Model does not have get_attention_patterns method")
            return None
        
        attention, time_labels = tft_model.get_attention_patterns(
            sample_index=sample_idx,
            aggregate=True
        )
        
        if attention.size == 0:
            LOGGER.warning("No attention data extracted")
            return None
        
        result = {
            'attention': attention,
            'time_labels': time_labels,
            'aggregated': True,
            'model_name': tft_model.name
        }
        
        LOGGER.info("TFT attention extraction complete")
        return result
        
    except Exception as e:
        LOGGER.error("Failed to extract TFT attention: %s", e)
        return None


def plot_tft_attention_heatmap(
    attention_result: Dict[str, Any],
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
    cmap: str = 'YlOrRd'
) -> Optional[plt.Figure]:
    """
    Create attention heatmap for TFT model.
    
    Args:
        attention_result: Output from extract_tft_attention
        output_path: Path to save plot
        figsize: Figure size in inches
        cmap: Colormap name
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_PLOTTING:
        LOGGER.warning("Plotting libraries not available")
        return None
    
    if attention_result is None:
        LOGGER.warning("No attention result provided")
        return None
    
    try:
        attention = attention_result['attention']
        time_labels = attention_result['time_labels']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attention,
            xticklabels=time_labels,
            yticklabels=[f"Forecast t+{i+1}" for i in range(attention.shape[0])],
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            linewidths=0.5,
            linecolor='white',
            annot=True if attention.shape[0] <= 12 else False,
            fmt='.3f' if attention.shape[0] <= 12 else None
        )
        
        ax.set_title(
            'TFT Temporal Attention Pattern\nWhich Historical Periods Drive Each Forecast?',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Historical Time Steps (Encoder)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Forecast Horizon (Decoder)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            LOGGER.info("TFT attention heatmap saved to %s", output_path)
        
        return fig
        
    except Exception as e:
        LOGGER.error("Failed to create TFT attention heatmap: %s", e)
        return None


def extract_tft_variable_importance(
    tft_model: Any,
    top_k: int = 20
) -> Optional[pd.DataFrame]:
    """
    Extract variable importance from TFT model.
    
    Args:
        tft_model: Fitted TFT model
        top_k: Number of top features to return
        
    Returns:
        DataFrame with variable importance rankings
    """
    LOGGER.info("Extracting variable importance from TFT model")
    
    try:
        if not hasattr(tft_model, 'get_variable_importance'):
            LOGGER.warning("Model does not have get_variable_importance method")
            return None
        
        importance_df = tft_model.get_variable_importance(top_k=top_k)
        
        LOGGER.info("TFT variable importance extracted: %d features", len(importance_df))
        return importance_df
        
    except Exception as e:
        LOGGER.error("Failed to extract TFT variable importance: %s", e)
        return None


# ============================================================================
# Unified Interpretation Report Generation
# ============================================================================

def generate_interpretation_report(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    output_dir: Path,
    model_type: str = "auto"
) -> Dict[str, Path]:
    """
    Generate comprehensive interpretation report for a model.
    
    Automatically detects model type and generates appropriate
    interpretability artifacts (SHAP for tree models, attention for TFT).
    
    Args:
        model: Fitted model
        X: Feature matrix
        y: Target variable
        model_name: Name of the model
        output_dir: Directory to save outputs
        model_type: 'tree', 'tft', or 'auto' (auto-detect)
        
    Returns:
        Dictionary mapping artifact names to file paths
    """
    LOGGER.info("Generating interpretation report for %s", model_name)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    
    # Auto-detect model type
    if model_type == "auto":
        model_class = model.__class__.__name__.lower()
        if any(keyword in model_class for keyword in ("xgb", "gradient", "forest", "boost")):
            model_type = "tree"
        elif any(keyword in model_class for keyword in ("lasso", "linear", "ridge", "elastic")):
            model_type = "linear"
        elif "tft" in model_class or "temporal" in model_class:
            model_type = "tft"
        else:
            model_type = "unknown"
        LOGGER.info("Auto-detected model type: %s", model_type)
    
    # Generate tree-based model interpretations
    if model_type == "tree":
        LOGGER.info("Generating SHAP interpretations for tree-based model")
        
        # Compute SHAP values
        shap_result = compute_shap_values(model, X, model_type="tree")
        
        if shap_result is not None:
            # Summary plot (beeswarm)
            summary_path = output_dir / f"{model_name}_shap_summary.png"
            plot_shap_summary(shap_result, summary_path, plot_type="dot")
            artifacts['shap_summary'] = summary_path
            
            # Bar plot
            bar_path = output_dir / f"{model_name}_shap_bar.png"
            plot_shap_summary(shap_result, bar_path, plot_type="bar")
            artifacts['shap_bar'] = bar_path
            
            # Waterfall for latest prediction
            waterfall_path = output_dir / f"{model_name}_shap_waterfall.png"
            plot_shap_waterfall(shap_result, sample_idx=-1, output_path=waterfall_path)
            artifacts['shap_waterfall'] = waterfall_path
            
            # Dependence plots for top 3 features
            if len(shap_result['feature_names']) > 0:
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_result['shap_values']).mean(axis=0)
                top_features = [shap_result['feature_names'][i] 
                              for i in np.argsort(mean_abs_shap)[-3:][::-1]]
                
                for i, feature in enumerate(top_features):
                    dep_path = output_dir / f"{model_name}_shap_dep_{i+1}_{feature}.png"
                    plot_shap_dependence(shap_result, feature, output_path=dep_path)
                    artifacts[f'shap_dependence_{i+1}'] = dep_path
    
    # Generate TFT interpretations
    elif model_type == "linear":
        LOGGER.info("Generating SHAP interpretations for linear model")
        shap_result = compute_shap_values(model, X, model_type="linear")

        if shap_result is not None:
            summary_path = output_dir / f"{model_name}_shap_summary.png"
            plot_shap_summary(shap_result, summary_path, plot_type="bar")
            artifacts['shap_summary'] = summary_path

            # Waterfall plot for linear models can still be informative
            waterfall_path = output_dir / f"{model_name}_shap_waterfall.png"
            plot_shap_waterfall(shap_result, sample_idx=-1, output_path=waterfall_path)
            artifacts['shap_waterfall'] = waterfall_path

            # Dependence plots for top features
            if len(shap_result['feature_names']) > 0:
                mean_abs_shap = np.abs(shap_result['shap_values']).mean(axis=0)
                top_features = [shap_result['feature_names'][i]
                                for i in np.argsort(mean_abs_shap)[-3:][::-1]]
                for i, feature in enumerate(top_features):
                    dep_path = output_dir / f"{model_name}_shap_dep_{i+1}_{feature}.png"
                    plot_shap_dependence(shap_result, feature, output_path=dep_path)
                    artifacts[f'shap_dependence_{i+1}'] = dep_path

    elif model_type == "tft":
        LOGGER.info("Generating TFT interpretations (attention & variable importance)")
        
        # Extract attention
        attention_result = extract_tft_attention(model, X)
        if attention_result is not None:
            attention_path = output_dir / f"{model_name}_attention_heatmap.png"
            plot_tft_attention_heatmap(attention_result, attention_path)
            artifacts['attention_heatmap'] = attention_path
        
        # Extract variable importance
        importance_df = extract_tft_variable_importance(model, top_k=20)
        if importance_df is not None:
            importance_path = output_dir / f"{model_name}_variable_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            artifacts['variable_importance'] = importance_path
            
            # Plot variable importance
            if hasattr(model, 'plot_variable_importance'):
                vi_plot_path = output_dir / f"{model_name}_variable_importance.png"
                model.plot_variable_importance(save_path=vi_plot_path, top_k=15)
                artifacts['variable_importance_plot'] = vi_plot_path
        
        # Generate full interpretation report if available
        if hasattr(model, 'create_interpretation_report'):
            try:
                model.create_interpretation_report(
                    output_dir=output_dir,
                    forecast_horizons=[1, 3, 6, 12]
                )
                artifacts['full_report'] = output_dir / "interpretation_report.md"
            except Exception as e:
                LOGGER.warning("Failed to generate full TFT report: %s", e)
    
    LOGGER.info("Interpretation report complete. Generated %d artifacts", len(artifacts))
    return artifacts


def create_model_comparison_report(
    backtest_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    include_interpretations: bool = True
) -> Path:
    """
    Create comprehensive comparison report across all models.
    
    Args:
        backtest_results: Dictionary mapping model names to backtest DataFrames
        output_dir: Directory to save report
        include_interpretations: Whether to include interpretation artifacts
        
    Returns:
        Path to generated markdown report
    """
    LOGGER.info("Creating model comparison report")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# OER Forecasting Model Comparison Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Performance Summary",
        ""
    ]
    
    # Performance table
    summary_data = []
    for model_name, results in backtest_results.items():
        summary = aggregate_backtest_results(results)
        if not summary.empty:
            row = {'Model': model_name.upper()}
            for col in summary.columns:
                row[col.upper()] = f"{summary[col].iloc[0]:.4f}"
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        report_lines.append("| " + " | ".join(summary_df.columns) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(summary_df.columns)) + " |")
        for _, row in summary_df.iterrows():
            report_lines.append("| " + " | ".join(row.values) + " |")
        report_lines.append("")
    
    # Add interpretation sections
    if include_interpretations:
        report_lines.extend([
            "---",
            "",
            "## Model Interpretations",
            ""
        ])
        
        for model_name in backtest_results.keys():
            report_lines.extend([
                f"### {model_name.upper()}",
                ""
            ])
            
            # Check for SHAP artifacts
            shap_summary = output_dir / f"{model_name}_shap_summary.png"
            if shap_summary.exists():
                report_lines.extend([
                    "#### SHAP Feature Importance",
                    "",
                    f"![SHAP Summary]({shap_summary.name})",
                    ""
                ])
            
            # Check for TFT artifacts
            attention_heatmap = output_dir / f"{model_name}_attention_heatmap.png"
            if attention_heatmap.exists():
                report_lines.extend([
                    "#### TFT Attention Patterns",
                    "",
                    f"![Attention Heatmap]({attention_heatmap.name})",
                    ""
                ])
            
            report_lines.append("---")
            report_lines.append("")
    
    # Write report
    report_path = output_dir / "model_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    LOGGER.info("Model comparison report saved to %s", report_path)
    return report_path
