"""
Enhanced OER Forecasting Dashboard for Bloomberg Terminal (BQNT)

This dashboard provides a comprehensive, professional-grade interface for
exploring OER forecasts, model performance, and interpretability insights.

Designed for Bloomberg Terminal integration and economist audiences.

Key Features:
1. Interactive multi-model forecast comparisons
2. Rolling backtest performance metrics
3. TFT interpretability visualizations (attention, variable importance)
4. Panel structure analysis
5. Economic narrative generation
6. Export-ready charts for presentations

Built with Plotly Dash for high interactivity and professional aesthetics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from ..config import AppConfig
from ..utils.io import read_dataframe
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Bloomberg Terminal color palette
BLOOMBERG_COLORS = {
    'primary': '#F58025',      # Bloomberg orange
    'secondary': '#000000',    # Black
    'background': '#0E0E0E',   # Dark background
    'text': '#FFFFFF',         # White text
    'grid': '#2A2A2A',         # Dark grid
    'positive': '#00C853',     # Green
    'negative': '#FF1744',     # Red
    'models': [
        '#F58025',  # Orange
        '#00BCD4',  # Cyan
        '#FFC107',  # Amber
        '#9C27B0',  # Purple
        '#4CAF50',  # Green
        '#FF5722',  # Deep Orange
    ]
}


class OERForecastDashboard:
    """
    Main dashboard class for OER forecasting visualization.
    
    This creates a Dash application with multiple pages:
    - Overview: Current forecasts and model comparison
    - Performance: Detailed backtest metrics and charts
    - Interpretation: TFT attention and variable importance
    - Data: Panel structure analysis and feature exploration
    """
    
    def __init__(
        self,
        config: AppConfig,
        app_title: str = "OER Forecast Dashboard - BQNT",
        host: str = "127.0.0.1",
        port: int = 8050
    ):
        self.config = config
        self.host = host
        self.port = port
        
        # Initialize Dash app with Bloomberg-inspired theme
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title=app_title,
            suppress_callback_exceptions=True
        )
        
        # Load data
        self._load_data()
        
        # Build layout
        self._build_layout()
        
        # Register callbacks
        self._register_callbacks()
        
    def _load_data(self):
        """Load all necessary data for the dashboard."""
        LOGGER.info("Loading data for dashboard")
        
        artifacts_dir = self.config.artifacts_dir
        
        # Load processed features
        try:
            self.features = read_dataframe(
                self.config.processed_dir / "features_processed.csv",
                parse_dates=[0]
            )
            self.features.index = pd.to_datetime(self.features.index)
        except Exception as e:
            LOGGER.warning("Could not load features: %s", e)
            self.features = pd.DataFrame()
        
        # Load backtest results for each model
        self.backtest_results = {}
        self.model_summaries = {}
        
        for model_file in artifacts_dir.glob("backtest_*.csv"):
            model_name = model_file.stem.replace("backtest_", "")
            try:
                df = read_dataframe(model_file, parse_dates=['timestamp'])
                self.backtest_results[model_name] = df
                LOGGER.info("Loaded backtest results for %s", model_name)
            except Exception as e:
                LOGGER.warning("Could not load backtest for %s: %s", model_name, e)
        
        for summary_file in artifacts_dir.glob("summary_*.csv"):
            model_name = summary_file.stem.replace("summary_", "")
            try:
                df = read_dataframe(summary_file)
                self.model_summaries[model_name] = df
            except Exception as e:
                LOGGER.warning("Could not load summary for %s: %s", model_name, e)
        
        # Load latest forecasts
        try:
            self.latest_forecasts = read_dataframe(
                artifacts_dir / "latest_forecasts.csv",
                parse_dates=[0]
            )
            self.latest_forecasts.index = pd.to_datetime(self.latest_forecasts.index)
        except Exception as e:
            LOGGER.warning("Could not load latest forecasts: %s", e)
            self.latest_forecasts = pd.DataFrame()
        
        # Load TFT interpretations if available
        interp_file = artifacts_dir / "tft_interpretation" / "explanations.json"
        if interp_file.exists():
            try:
                with open(interp_file, 'r') as f:
                    self.tft_interpretations = json.load(f)
            except Exception as e:
                LOGGER.warning("Could not load TFT interpretations: %s", e)
                self.tft_interpretations = {}
        else:
            self.tft_interpretations = {}
    
    def _build_layout(self):
        """Construct the dashboard layout."""
        
        # Header
        header = dbc.Navbar(
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.H2("OER Forecast Dashboard", className="text-warning mb-0"),
                            html.P("Owners' Equivalent Rent - Bloomberg Terminal Analytics",
                                  className="text-muted mb-0 small")
                        ]),
                        width="auto"
                    ),
                ], align="center", className="w-100"),
            ], fluid=True),
            color="dark",
            dark=True,
            className="mb-4"
        )
        
        # Navigation tabs
        tabs = dbc.Tabs([
            dbc.Tab(label="Overview", tab_id="tab-overview", label_style={"color": BLOOMBERG_COLORS['text']}),
            dbc.Tab(label="Performance", tab_id="tab-performance", label_style={"color": BLOOMBERG_COLORS['text']}),
            dbc.Tab(label="Interpretation", tab_id="tab-interpretation", label_style={"color": BLOOMBERG_COLORS['text']}),
            dbc.Tab(label="Data Explorer", tab_id="tab-data", label_style={"color": BLOOMBERG_COLORS['text']}),
        ], id="tabs", active_tab="tab-overview", className="mb-4")
        
        # Content area (will be populated by callbacks)
        content = html.Div(id="tab-content", className="p-3")
        
        # Main layout
        self.app.layout = dbc.Container([
            header,
            tabs,
            content
        ], fluid=True, style={"backgroundColor": BLOOMBERG_COLORS['background']})
    
    def _register_callbacks(self):
        """Register all interactive callbacks."""
        
        @self.app.callback(
            Output("tab-content", "children"),
            Input("tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            if active_tab == "tab-overview":
                return self._create_overview_tab()
            elif active_tab == "tab-performance":
                return self._create_performance_tab()
            elif active_tab == "tab-interpretation":
                return self._create_interpretation_tab()
            elif active_tab == "tab-data":
                return self._create_data_tab()
            return html.Div("Select a tab")
    
    def _create_overview_tab(self) -> dbc.Container:
        """Create the overview tab with current forecasts."""
        
        # Key metrics cards
        metrics = self._calculate_current_metrics()
        
        metric_cards = dbc.Row([
            dbc.Col(self._create_metric_card(
                "Current OER",
                f"{metrics['current_oer']:.2f}%",
                metrics['oer_change'],
                "YoY change"
            ), width=3),
            dbc.Col(self._create_metric_card(
                "Best Model",
                metrics['best_model'],
                metrics['best_model_rmse'],
                "RMSE on validation"
            ), width=3),
            dbc.Col(self._create_metric_card(
                "12M Forecast",
                f"{metrics['forecast_12m']:.2f}%",
                None,
                "Consensus estimate"
            ), width=3),
            dbc.Col(self._create_metric_card(
                "Forecast Range",
                f"{metrics['forecast_range']:.2f}%",
                None,
                "Model dispersion"
            ), width=3),
        ], className="mb-4")
        
        # Main forecast chart
        forecast_chart = dcc.Graph(
            figure=self._create_forecast_comparison_chart(),
            config={'displayModeBar': False},
            style={'height': '500px'}
        )
        
        # Model comparison table
        comparison_table = self._create_model_comparison_table()
        
        return dbc.Container([
            html.H3("Forecast Overview", className="text-warning mb-4"),
            metric_cards,
            dbc.Card([
                dbc.CardHeader("Multi-Model Forecast Comparison"),
                dbc.CardBody(forecast_chart)
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Model Performance Comparison"),
                dbc.CardBody(comparison_table)
            ])
        ], fluid=True)
    
    def _create_performance_tab(self) -> dbc.Container:
        """Create the performance analysis tab."""
        
        # Model selector
        model_options = [{'label': name, 'value': name} 
                        for name in self.backtest_results.keys()]
        
        model_selector = dbc.Row([
            dbc.Col([
                dbc.Label("Select Model:"),
                dcc.Dropdown(
                    id='model-selector',
                    options=model_options,
                    value=model_options[0]['value'] if model_options else None,
                    className="mb-3"
                )
            ], width=4)
        ])
        
        # Performance charts will be generated by callback
        charts = html.Div(id='performance-charts')
        
        return dbc.Container([
            html.H3("Backtest Performance Analysis", className="text-warning mb-4"),
            model_selector,
            charts
        ], fluid=True)
    
    def _create_interpretation_tab(self) -> dbc.Container:
        """Create the TFT interpretation tab."""
        
        if not self.tft_interpretations:
            return dbc.Container([
                html.H3("Model Interpretation", className="text-warning mb-4"),
                dbc.Alert(
                    "TFT interpretation data not available. Run TFT model training and interpretation generation.",
                    color="warning"
                )
            ], fluid=True)
        
        # Horizon selector
        horizon_options = [{'label': f"{h} Month(s)", 'value': h}
                          for h in sorted(self.tft_interpretations.keys())]
        
        horizon_selector = dbc.Row([
            dbc.Col([
                dbc.Label("Forecast Horizon:"),
                dcc.Dropdown(
                    id='horizon-selector',
                    options=horizon_options,
                    value=horizon_options[0]['value'] if horizon_options else None,
                    className="mb-3"
                )
            ], width=4)
        ])
        
        # Interpretation content
        interp_content = html.Div(id='interpretation-content')
        
        return dbc.Container([
            html.H3("TFT Model Interpretation", className="text-warning mb-4"),
            html.P("Understanding what drives the forecast through attention and variable importance",
                  className="text-muted mb-4"),
            horizon_selector,
            interp_content
        ], fluid=True)
    
    def _create_data_tab(self) -> dbc.Container:
        """Create the data exploration tab."""
        
        if self.features.empty:
            return dbc.Container([
                html.H3("Data Explorer", className="text-warning mb-4"),
                dbc.Alert("Feature data not available.", color="warning")
            ], fluid=True)
        
        # Feature selector
        feature_options = [{'label': col, 'value': col} 
                          for col in self.features.columns]
        
        feature_selector = dbc.Row([
            dbc.Col([
                dbc.Label("Select Features to Display:"),
                dcc.Dropdown(
                    id='feature-selector',
                    options=feature_options,
                    value=feature_options[:5] if len(feature_options) >= 5 else [f['value'] for f in feature_options],
                    multi=True,
                    className="mb-3"
                )
            ], width=8)
        ])
        
        # Time series chart
        time_series_chart = html.Div(id='time-series-chart')
        
        # Correlation heatmap
        correlation_chart = dcc.Graph(
            figure=self._create_correlation_heatmap(),
            config={'displayModeBar': False},
            style={'height': '600px'}
        )
        
        return dbc.Container([
            html.H3("Data Explorer", className="text-warning mb-4"),
            feature_selector,
            dbc.Card([
                dbc.CardHeader("Feature Time Series"),
                dbc.CardBody(time_series_chart)
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Feature Correlation Matrix"),
                dbc.CardBody(correlation_chart)
            ])
        ], fluid=True)
    
    def _create_metric_card(
        self,
        title: str,
        value: str,
        change: Optional[float],
        subtitle: str
    ) -> dbc.Card:
        """Create a metric display card."""
        
        change_badge = None
        if change is not None:
            color = "success" if change >= 0 else "danger"
            icon = "↑" if change >= 0 else "↓"
            change_badge = dbc.Badge(
                f"{icon} {abs(change):.2f}",
                color=color,
                className="ms-2"
            )
        
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted mb-2"),
                html.Div([
                    html.H3(value, className="mb-0 d-inline"),
                    change_badge if change_badge else html.Span()
                ]),
                html.Small(subtitle, className="text-muted")
            ])
        ], className="h-100")
    
    def _calculate_current_metrics(self) -> Dict:
        """Calculate current dashboard metrics."""
        metrics = {
            'current_oer': 0.0,
            'oer_change': 0.0,
            'best_model': 'N/A',
            'best_model_rmse': 0.0,
            'forecast_12m': 0.0,
            'forecast_range': 0.0
        }
        
        # Current OER from features
        if not self.features.empty:
            target_col = self.config.features.get('target_column')
            if target_col and target_col in self.features.columns:
                metrics['current_oer'] = self.features[target_col].iloc[-1]
                if len(self.features) >= 12:
                    metrics['oer_change'] = (
                        self.features[target_col].iloc[-1] - 
                        self.features[target_col].iloc[-12]
                    )
        
        # Best model from summaries
        if self.model_summaries:
            best_rmse = float('inf')
            for model_name, summary in self.model_summaries.items():
                if 'rmse' in summary.columns:
                    rmse = summary['rmse'].iloc[0]
                    if rmse < best_rmse:
                        best_rmse = rmse
                        metrics['best_model'] = model_name
                        metrics['best_model_rmse'] = rmse
        
        # Forecast metrics
        if not self.latest_forecasts.empty and len(self.latest_forecasts) >= 12:
            forecasts_12m = self.latest_forecasts.iloc[11]
            metrics['forecast_12m'] = forecasts_12m.mean()
            metrics['forecast_range'] = forecasts_12m.std()
        
        return metrics
    
    def _create_forecast_comparison_chart(self) -> go.Figure:
        """Create multi-model forecast comparison chart."""
        
        fig = go.Figure()
        
        # Plot historical data
        if not self.features.empty:
            target_col = self.config.features.get('target_column')
            if target_col and target_col in self.features.columns:
                fig.add_trace(go.Scatter(
                    x=self.features.index,
                    y=self.features[target_col],
                    mode='lines',
                    name='Actual OER',
                    line=dict(color=BLOOMBERG_COLORS['text'], width=2)
                ))
        
        # Plot forecasts from each model
        if not self.latest_forecasts.empty:
            for i, col in enumerate(self.latest_forecasts.columns):
                color = BLOOMBERG_COLORS['models'][i % len(BLOOMBERG_COLORS['models'])]
                fig.add_trace(go.Scatter(
                    x=self.latest_forecasts.index,
                    y=self.latest_forecasts[col],
                    mode='lines+markers',
                    name=f'{col} forecast',
                    line=dict(color=color, width=2, dash='dash')
                ))
        
        # Styling
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=BLOOMBERG_COLORS['background'],
            paper_bgcolor=BLOOMBERG_COLORS['background'],
            font=dict(color=BLOOMBERG_COLORS['text']),
            title="OER Forecast: Multi-Model Comparison",
            xaxis_title="Date",
            yaxis_title="OER YoY Change (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_model_comparison_table(self) -> dbc.Table:
        """Create model performance comparison table."""
        
        if not self.model_summaries:
            return html.P("No model summaries available", className="text-muted")
        
        # Collect metrics from all models
        rows = []
        for model_name, summary in self.model_summaries.items():
            row_data = {'Model': model_name.upper()}
            for col in summary.columns:
                if col != 'model':
                    row_data[col.upper()] = f"{summary[col].iloc[0]:.4f}"
            rows.append(row_data)
        
        if not rows:
            return html.P("No metrics available", className="text-muted")
        
        # Create table
        df = pd.DataFrame(rows)
        
        return dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            dark=True,
            className="mb-0"
        )
    
    def _create_correlation_heatmap(self) -> go.Figure:
        """Create feature correlation heatmap."""
        
        if self.features.empty:
            return go.Figure()
        
        # Calculate correlation matrix
        corr = self.features.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=BLOOMBERG_COLORS['background'],
            paper_bgcolor=BLOOMBERG_COLORS['background'],
            font=dict(color=BLOOMBERG_COLORS['text']),
            title="Feature Correlation Matrix",
            xaxis=dict(tickangle=-45),
            height=600
        )
        
        return fig
    
    def run(self, debug: bool = False):
        """Start the dashboard server."""
        LOGGER.info("Starting OER Forecast Dashboard on http://%s:%d", self.host, self.port)
        print(f"\n{'='*70}")
        print(f"OER FORECAST DASHBOARD")
        print(f"{'='*70}")
        print(f"Dashboard URL: http://{self.host}:{self.port}")
        print(f"Press Ctrl+C to stop the server")
        print(f"{'='*70}\n")
        
        self.app.run_server(host=self.host, port=self.port, debug=debug)


def create_dashboard(config: AppConfig, **kwargs) -> OERForecastDashboard:
    """
    Factory function to create and configure the dashboard.
    
    Args:
        config: Application configuration
        **kwargs: Additional arguments passed to OERForecastDashboard
        
    Returns:
        Configured dashboard instance
    """
    return OERForecastDashboard(config, **kwargs)
