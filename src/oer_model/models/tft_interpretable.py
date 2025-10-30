"""
Enhanced Temporal Fusion Transformer with Full Interpretability

This module extends the basic TFT implementation with comprehensive
interpretability features essential for economic forecasting.

Key Features for Economists:
1. Variable importance extraction - which economic indicators matter most
2. Attention visualization - which historical periods drive forecasts
3. Quantile predictions - uncertainty quantification
4. Narrative generation - automatic explanation of forecasts

Reference:
Lim et al. (2021) "Temporal Fusion Transformers for Interpretable 
Multi-horizon Time Series Forecasting" - Google Research
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .tft import TemporalFusionTransformerModel, TFTParams
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

try:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_INTERPRETATION_LIBS = True
except ImportError:
    HAS_INTERPRETATION_LIBS = False


class InterpretableTFT(TemporalFusionTransformerModel):
    """
    TFT model with enhanced interpretability methods.
    
    This class extends the base TFT with methods to extract and visualize
    the model's internal decision-making process, making it suitable for
    presentation to economists and policymakers.
    """
    
    def __init__(
        self,
        name: str = "interpretable_tft",
        params: Optional[TFTParams | Dict[str, Any]] = None,
        known_future_covariates: Optional[List[str]] = None
    ):
        super().__init__(name, params, known_future_covariates)
        
        # Storage for interpretability outputs
        self.variable_importance_: Optional[pd.DataFrame] = None
        self.attention_patterns_: Optional[Dict[str, np.ndarray]] = None
        
    def get_variable_importance(
        self,
        top_k: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Extract and rank variable importance from the trained TFT model.
        
        The TFT uses Variable Selection Networks that learn which features
        are most relevant. This method extracts those learned importances.
        
        Args:
            top_k: Number of top features to return
            normalize: Whether to normalize importances to sum to 1
            
        Returns:
            DataFrame with columns ['variable', 'importance', 'rank']
            
        Example:
            >>> model.fit(X, y)
            >>> importance = model.get_variable_importance(top_k=10)
            >>> print(importance)
            variable                    importance  rank
            zori_yoy_lag12m            0.234       1
            zhvi_yoy_lag16m            0.189       2
            unemployment_rate          0.156       3
            ...
        """
        if self._model is None or self._dataset is None:
            raise RuntimeError("Model must be fitted before extracting importance")
        
        if not HAS_INTERPRETATION_LIBS:
            LOGGER.warning("Interpretation libraries not available")
            return pd.DataFrame()
        
        LOGGER.info("Extracting variable importance from TFT Variable Selection Networks")
        
        try:
            # Get a sample batch to run interpretation on
            dataloader = self._dataset.to_dataloader(train=False, batch_size=1)
            x_sample = next(iter(dataloader))
            
            # Forward pass with interpretation
            with torch.no_grad():
                interpretation = self._model.interpret_output(
                    x_sample,
                    output=self._model(x_sample)
                )
            
            # Extract variable selection weights
            # TFT has separate importance for encoder and decoder variables
            encoder_importance = interpretation.get('encoder_variables', torch.tensor([]))
            decoder_importance = interpretation.get('decoder_variables', torch.tensor([]))
            
            # Get variable names
            encoder_vars = self._dataset.encoder_variables
            decoder_vars = self._dataset.decoder_variables
            
            # Convert to numpy and combine
            if len(encoder_importance) > 0:
                enc_imp = encoder_importance.cpu().numpy()
            else:
                enc_imp = np.array([])
                
            if len(decoder_importance) > 0:
                dec_imp = decoder_importance.cpu().numpy()
            else:
                dec_imp = np.array([])
            
            # Create DataFrame
            all_vars = list(encoder_vars) + list(decoder_vars)
            all_importance = np.concatenate([enc_imp, dec_imp]) if len(enc_imp) > 0 and len(dec_imp) > 0 else enc_imp
            
            importance_df = pd.DataFrame({
                'variable': all_vars,
                'importance': all_importance
            })
            
            # Normalize if requested
            if normalize and len(importance_df) > 0:
                importance_df['importance'] = (
                    importance_df['importance'] / importance_df['importance'].sum()
                )
            
            # Sort and rank
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            # Store for later use
            self.variable_importance_ = importance_df
            
            LOGGER.info("Top 5 important variables: %s", 
                       importance_df.head(5)['variable'].tolist())
            
            return importance_df.head(top_k)
            
        except Exception as e:
            LOGGER.error("Failed to extract variable importance: %s", e)
            return pd.DataFrame(columns=['variable', 'importance', 'rank'])
    
    def get_attention_patterns(
        self,
        sample_index: Optional[int] = None,
        aggregate: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights showing temporal focus.
        
        The multi-head attention mechanism in TFT learns which historical
        time periods are most relevant for each forecast horizon. This
        method extracts and visualizes those patterns.
        
        Args:
            sample_index: Which validation sample to use (None = last)
            aggregate: Whether to average across attention heads
            
        Returns:
            Tuple of (attention_array, time_labels)
            - attention_array: shape (n_decoder, n_encoder) if aggregate=True
                             or (n_heads, n_decoder, n_encoder) if False
            - time_labels: List of strings labeling encoder timesteps
            
        Example:
            High attention at "t-12" for forecast "t+1" means the model
            considers the value 12 months ago crucial for next month's forecast.
        """
        if self._model is None or self._dataset is None:
            raise RuntimeError("Model must be fitted before extracting attention")
        
        if not HAS_INTERPRETATION_LIBS:
            LOGGER.warning("Interpretation libraries not available")
            return np.array([]), []
        
        LOGGER.info("Extracting attention patterns from TFT")
        
        try:
            # Get validation dataloader
            dataloader = self._dataset.to_dataloader(train=False, batch_size=1)
            
            # Get specific sample or last one
            if sample_index is not None:
                for i, batch in enumerate(dataloader):
                    if i == sample_index:
                        x_sample = batch
                        break
            else:
                # Get last batch
                for batch in dataloader:
                    x_sample = batch
            
            # Forward pass with attention output
            with torch.no_grad():
                interpretation = self._model.interpret_output(
                    x_sample,
                    output=self._model(x_sample)
                )
            
            # Extract attention weights
            # Shape: (batch, n_heads, decoder_len, encoder_len)
            attention = interpretation.get('attention', None)
            
            if attention is None:
                LOGGER.warning("No attention found in model interpretation")
                return np.array([]), []
            
            # Convert to numpy and remove batch dimension
            attention_np = attention.cpu().numpy()[0]  # Shape: (n_heads, decoder_len, encoder_len)
            
            # Aggregate across heads if requested
            if aggregate:
                attention_np = attention_np.mean(axis=0)  # Shape: (decoder_len, encoder_len)
            
            # Create time labels
            encoder_length = self.params.max_encoder_length
            time_labels = [f"t-{encoder_length - i}" for i in range(encoder_length)]
            
            # Store patterns
            self.attention_patterns_ = {
                'attention': attention_np,
                'time_labels': time_labels,
                'aggregated': aggregate
            }
            
            return attention_np, time_labels
            
        except Exception as e:
            LOGGER.error("Failed to extract attention patterns: %s", e)
            return np.array([]), []
    
    def plot_variable_importance(
        self,
        save_path: Optional[Path] = None,
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Optional[plt.Figure]:
        """
        Create publication-quality variable importance plot.
        
        Args:
            save_path: Path to save figure (PNG, PDF, etc.)
            top_k: Number of top features to display
            figsize: Figure size in inches
            
        Returns:
            matplotlib Figure object (if plotting successful)
        """
        if not HAS_INTERPRETATION_LIBS:
            LOGGER.warning("matplotlib not available")
            return None
        
        importance_df = self.get_variable_importance(top_k=top_k)
        
        if importance_df.empty:
            LOGGER.warning("No variable importance data available")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(
            importance_df['variable'],
            importance_df['importance'],
            color='steelblue',
            edgecolor='navy',
            alpha=0.7
        )
        
        # Formatting
        ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title(
            'TFT Variable Importance\nKey Drivers of OER Forecast',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.invert_yaxis()  # Top = most important
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            LOGGER.info("Variable importance plot saved to %s", save_path)
        
        return fig
    
    def plot_attention_heatmap(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = 'YlOrRd'
    ) -> Optional[plt.Figure]:
        """
        Create attention heatmap showing temporal dependencies.
        
        This visualization is particularly powerful for explaining to
        economists WHY the model made a specific forecast - it shows
        which historical periods the model focused on.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size in inches
            cmap: Colormap name
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_INTERPRETATION_LIBS:
            LOGGER.warning("matplotlib/seaborn not available")
            return None
        
        attention, time_labels = self.get_attention_patterns(aggregate=True)
        
        if attention.size == 0:
            LOGGER.warning("No attention data available")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attention,
            xticklabels=time_labels,
            yticklabels=[f"Forecast t+{i+1}" for i in range(attention.shape[0])],
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            linewidths=0.5,
            linecolor='white',
            annot=True if attention.shape[0] <= 12 else False,  # Annotate if small
            fmt='.3f' if attention.shape[0] <= 12 else None
        )
        
        # Formatting
        ax.set_title(
            'TFT Temporal Attention Pattern\nWhich Historical Periods Drive Each Forecast?',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Historical Time Steps (Encoder)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Forecast Horizon (Decoder)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            LOGGER.info("Attention heatmap saved to %s", save_path)
        
        return fig
    
    def explain_forecast(
        self,
        forecast_horizon: int = 1,
        top_k_features: int = 5,
        top_k_timesteps: int = 3
    ) -> Dict[str, Any]:
        """
        Generate comprehensive forecast explanation for economists.
        
        This combines variable importance and attention patterns into a
        structured explanation that can be converted to narrative text.
        
        Args:
            forecast_horizon: Which forecast step to explain (1-indexed)
            top_k_features: Number of features to highlight
            top_k_timesteps: Number of time periods to highlight
            
        Returns:
            Dictionary with structured explanation:
            {
                'forecast_horizon': int,
                'top_features': List of top contributing features,
                'top_timesteps': List of most attended historical periods,
                'narrative': Human-readable explanation string,
                'feature_contributions': Detailed breakdown
            }
            
        Example output narrative:
            "The 1-month ahead OER forecast is primarily driven by zori_yoy_lag12m 
            (importance: 0.234). The model focuses most heavily on conditions 
            12 months ago (attention: 0.187), suggesting that last year's rental 
            market dynamics are key to understanding near-term OER movements."
        """
        # Get variable importance
        importance = self.get_variable_importance(top_k=top_k_features)
        
        # Get attention patterns
        attention, time_labels = self.get_attention_patterns(aggregate=True)
        
        if attention.size == 0 or importance.empty:
            return {
                'forecast_horizon': forecast_horizon,
                'error': 'Insufficient data for explanation'
            }
        
        # Extract attention for the specified forecast horizon
        horizon_idx = forecast_horizon - 1
        if horizon_idx >= attention.shape[0]:
            horizon_idx = attention.shape[0] - 1
            LOGGER.warning("Requested horizon %d exceeds model capacity, using %d",
                          forecast_horizon, horizon_idx + 1)
        
        horizon_attention = attention[horizon_idx, :]
        
        # Find most attended timesteps
        top_timestep_indices = np.argsort(horizon_attention)[-top_k_timesteps:][::-1]
        top_timesteps = [
            {
                'timestep': time_labels[i],
                'attention': float(horizon_attention[i]),
                'description': self._describe_timestep(time_labels[i])
            }
            for i in top_timestep_indices
        ]
        
        # Generate narrative
        narrative = self._generate_narrative(
            forecast_horizon,
            importance.to_dict('records')[:top_k_features],
            top_timesteps
        )
        
        explanation = {
            'forecast_horizon': forecast_horizon,
            'horizon_description': f"{forecast_horizon} month(s) ahead",
            'top_features': importance.to_dict('records')[:top_k_features],
            'top_timesteps': top_timesteps,
            'narrative': narrative,
            'model_confidence': self._estimate_confidence(horizon_attention)
        }
        
        return explanation
    
    def _describe_timestep(self, timestep_label: str) -> str:
        """Convert timestep label (e.g., 't-12') to description."""
        if timestep_label == 't-0':
            return 'current period'
        months_ago = int(timestep_label.split('-')[1])
        if months_ago == 1:
            return 'last month'
        elif months_ago == 12:
            return 'one year ago'
        elif months_ago == 24:
            return 'two years ago'
        else:
            return f'{months_ago} months ago'
    
    def _estimate_confidence(self, attention_weights: np.ndarray) -> str:
        """Estimate model confidence based on attention entropy."""
        # Calculate entropy of attention distribution
        attention_norm = attention_weights / attention_weights.sum()
        entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-10))
        max_entropy = np.log(len(attention_weights))
        normalized_entropy = entropy / max_entropy
        
        if normalized_entropy < 0.3:
            return 'High (focused attention)'
        elif normalized_entropy < 0.7:
            return 'Medium (distributed attention)'
        else:
            return 'Low (diffuse attention)'
    
    def _generate_narrative(
        self,
        horizon: int,
        top_features: List[Dict],
        top_timesteps: List[Dict]
    ) -> str:
        """Generate human-readable narrative explanation."""
        if not top_features or not top_timesteps:
            return "Insufficient data for narrative generation."
        
        # Extract top feature and timestep
        primary_feature = top_features[0]
        primary_timestep = top_timesteps[0]
        
        # Build narrative
        narrative_parts = []
        
        # Introduce forecast horizon
        narrative_parts.append(
            f"For the {horizon}-month ahead forecast of OER:"
        )
        
        # Feature importance
        narrative_parts.append(
            f"\n\n**Key Driver**: '{primary_feature['variable']}' is the most important "
            f"predictor (importance score: {primary_feature['importance']:.3f}). "
        )
        
        # Add context for top features
        if len(top_features) > 1:
            other_features = ", ".join([f"'{f['variable']}'" for f in top_features[1:3]])
            narrative_parts.append(
                f"Other significant factors include {other_features}."
            )
        
        # Temporal attention
        narrative_parts.append(
            f"\n\n**Temporal Focus**: The model pays strongest attention to "
            f"{primary_timestep['description']} (attention weight: "
            f"{primary_timestep['attention']:.3f}). "
        )
        
        # Add context for other timesteps
        if len(top_timesteps) > 1:
            other_times = " and ".join([
                ts['description'] for ts in top_timesteps[1:3]
            ])
            narrative_parts.append(
                f"The model also considers {other_times} when making this prediction."
            )
        
        # Economic interpretation
        narrative_parts.append(
            f"\n\n**Economic Interpretation**: This pattern suggests that "
            f"{'recent' if primary_timestep['timestep'].endswith(('-1', '-2', '-3')) else 'historical'} "
            f"conditions are particularly relevant for this forecast horizon. "
        )
        
        if 'zori' in primary_feature['variable'].lower() or 'zhvi' in primary_feature['variable'].lower():
            narrative_parts.append(
                f"The strong importance of market-based housing indicators reflects "
                f"the known lead-lag relationship between private rent indices and "
                f"the BLS's panel-surveyed OER statistic."
            )
        elif 'unemployment' in primary_feature['variable'].lower() or 'wage' in primary_feature['variable'].lower():
            narrative_parts.append(
                f"The prominence of labor market conditions highlights their role "
                f"in driving housing demand and households' ability to pay rent."
            )
        
        return "".join(narrative_parts)
    
    def create_interpretation_report(
        self,
        output_dir: Path,
        forecast_horizons: Optional[List[int]] = None
    ) -> Path:
        """
        Generate a complete interpretation report with all visualizations.
        
        This creates a comprehensive report suitable for presentation to
        stakeholders, including all plots and narrative explanations.
        
        Args:
            output_dir: Directory to save report files
            forecast_horizons: List of horizons to explain (default: [1, 3, 6, 12])
            
        Returns:
            Path to the generated report directory
        """
        if forecast_horizons is None:
            forecast_horizons = [1, 3, 6, 12]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("Generating TFT interpretation report in %s", output_dir)
        
        # Create visualizations
        self.plot_variable_importance(
            save_path=output_dir / "variable_importance.png",
            top_k=15
        )
        
        self.plot_attention_heatmap(
            save_path=output_dir / "attention_heatmap.png"
        )
        
        # Generate explanations for each horizon
        explanations = {}
        for h in forecast_horizons:
            if h <= self.params.max_prediction_length:
                explanations[h] = self.explain_forecast(
                    forecast_horizon=h,
                    top_k_features=10,
                    top_k_timesteps=5
                )
        
        # Save explanations as JSON
        import json
        with open(output_dir / "explanations.json", 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        
        # Create markdown report
        report_md = self._create_markdown_report(explanations)
        with open(output_dir / "interpretation_report.md", 'w') as f:
            f.write(report_md)
        
        LOGGER.info("Interpretation report complete. Files saved to %s", output_dir)
        
        return output_dir
    
    def _create_markdown_report(self, explanations: Dict[int, Dict]) -> str:
        """Generate markdown-formatted interpretation report."""
        lines = [
            "# TFT Model Interpretation Report",
            "",
            "## Owners' Equivalent Rent Forecast Explanation",
            "",
            f"**Model**: {self.name}",
            f"**Encoder Length**: {self.params.max_encoder_length} months",
            f"**Prediction Length**: {self.params.max_prediction_length} months",
            "",
            "---",
            "",
            "## Variable Importance",
            "",
            "![Variable Importance](variable_importance.png)",
            "",
            "The chart above shows which economic indicators the model considers",
            "most important when forecasting OER. Variables with higher importance",
            "scores have greater influence on the model's predictions.",
            "",
            "---",
            "",
            "## Temporal Attention Patterns",
            "",
            "![Attention Heatmap](attention_heatmap.png)",
            "",
            "The attention heatmap reveals which historical time periods the model",
            "focuses on for each forecast horizon. Darker colors indicate higher",
            "attention weights, meaning those periods are more influential.",
            "",
            "---",
            "",
            "## Forecast Explanations",
            ""
        ]
        
        for horizon, explanation in sorted(explanations.items()):
            lines.append(f"### {horizon}-Month Ahead Forecast")
            lines.append("")
            lines.append(explanation.get('narrative', 'No narrative available'))
            lines.append("")
            lines.append(f"**Model Confidence**: {explanation.get('model_confidence', 'Unknown')}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
