"""Interactive dashboard for forecasts and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

from ..evaluation.reporting import aggregate_backtest_results, prepare_dashboard_frame
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_backtests(artifacts_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for csv_path in artifacts_dir.glob("backtest_*.csv"):
        model_name = csv_path.stem.replace("backtest_", "")
        frames[model_name] = pd.read_csv(csv_path, parse_dates=["timestamp", "window_train_end"])
    return frames


def _build_layout(backtests: Dict[str, pd.DataFrame], forecasts: pd.DataFrame) -> html.Div:
    if forecasts.empty:
        forecast_fig = px.line(title="Model Forecasts")
    else:
        forecast_fig = px.line(
            forecasts,
            x=forecasts.index,
            y=forecasts.columns,
            title="Model Forecasts",
        )

    scorecards = []
    for name, frame in backtests.items():
        agg = aggregate_backtest_results(frame)
        if name in agg.index:
            summary_row = agg.loc[name]
        else:
            summary_row = agg.iloc[0]
        card = dbc.Card([
            dbc.CardHeader(name.upper()),
            dbc.CardBody([
                html.P(f"RMSE: {summary_row['rmse']:.4f}" if 'rmse' in agg.columns else "RMSE: n/a"),
                html.P(f"MAE: {summary_row['mae']:.4f}" if 'mae' in agg.columns else "MAE: n/a"),
                html.P(f"MAPE: {summary_row['mape']:.2f}%" if 'mape' in agg.columns else "MAPE: n/a"),
            ]),
        ])
        scorecards.append(card)

    backtest_frames = [frame.assign(model=name) for name, frame in backtests.items()]
    if backtest_frames:
        comparison_frame = pd.concat(backtest_frames)
        comparison_pivot = prepare_dashboard_frame(comparison_frame)
        comparison_fig = px.line(
            comparison_pivot,
            x=comparison_pivot.index,
            y=comparison_pivot.columns,
            title="Backtest Predictions vs Actuals",
        )
    else:
        comparison_fig = px.line(title="Backtest Predictions vs Actuals")

    return html.Div([
        dbc.Container([
            html.H2("OER Forecasting Dashboard"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=forecast_fig), width=12),
            ]),
            html.H3("Model Scorecards"),
            dbc.Row([dbc.Col(card, width=3) for card in scorecards]),
            html.H3("Backtest Comparison"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=comparison_fig), width=12),
            ]),
        ], fluid=True),
    ])


def create_dashboard(artifacts_dir: Path, forecasts_path: Path) -> Dash:
    """Create the Dash application."""
    backtests = _load_backtests(artifacts_dir)
    if forecasts_path.exists():
        forecasts = pd.read_csv(forecasts_path, index_col=0, parse_dates=True)
    else:
        LOGGER.warning("Forecast file %s not found; creating placeholder", forecasts_path)
        forecasts = pd.DataFrame()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = _build_layout(backtests, forecasts)
    return app
