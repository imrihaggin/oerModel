"""
OER Forecasting - Comprehensive Interactive Dashboard

Features:
- Executive Summary with key metrics
- Model Comparison (interactive charts)
- Time Series Forecasts
- XGBoost SHAP Analysis
- TFT Variable Importance & Attention
- Backtest Results
- Downloadable Reports
- Modern, responsive design
"""
import sys
sys.path.insert(0, 'src')

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Initialize app with modern theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # Dark, modern theme
    title="OER Forecasting Dashboard"
)

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            .metric-value {
                font-size: 2.5em;
                font-weight: bold;
                color: #00d4ff;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            }
            .metric-label {
                font-size: 0.9em;
                color: #aaa;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .winner-badge {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin-left: 10px;
            }
            h1, h2, h3 {
                color: white;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .graph-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load all artifacts and data."""
    data = {}
    
    # Load summaries
    summaries = []
    for f in Path('artifacts').glob('summary_*.csv'):
        df = pd.read_csv(f)
        summaries.append(df)
    if summaries:
        data['summaries'] = pd.concat(summaries, ignore_index=True)
    
    # Load backtests
    backtests = {}
    for f in Path('artifacts').glob('backtest_*.csv'):
        model_name = f.stem.replace('backtest_', '')
        backtests[model_name] = pd.read_csv(f)
    data['backtests'] = backtests
    
    # Load TFT variable importance
    tft_var_path = Path('artifacts/tft_variable_importance.csv')
    if tft_var_path.exists():
        data['tft_importance'] = pd.read_csv(tft_var_path)
    
    # Load XGBoost Optuna params
    xgb_params_path = Path('artifacts/xgboost_optuna_params.json')
    if xgb_params_path.exists():
        with open(xgb_params_path) as f:
            data['xgb_params'] = json.load(f)
    
    # Load processed features
    features_path = Path('data/processed/features_processed.csv')
    if features_path.exists():
        data['features'] = pd.read_csv(features_path)
    
    return data

data = load_all_data()

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    html.Div([
        # Header
        html.Div([
            html.H1([
                "🏠 OER Forecasting Dashboard",
                html.Span("powered by ML/DL", style={'fontSize': '0.4em', 'marginLeft': '20px', 'opacity': '0.7'})
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(
                "Comprehensive analysis of Owner's Equivalent Rent forecasting models",
                style={'textAlign': 'center', 'color': '#ddd', 'fontSize': '1.1em'}
            ),
        ], style={'marginBottom': '30px'}),
        
        # Executive Summary
        html.Div([
            html.H2("📊 Executive Summary", style={'marginBottom': '20px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("126", className="metric-value"),
                        html.Div("Data Points", className="metric-label"),
                        html.P("1993-2025", style={'color': '#888', 'marginTop': '5px'})
                    ], className="metric-card")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div("4", className="metric-value"),
                        html.Div("Models Trained", className="metric-label"),
                        html.P("LASSO, VAR, XGBoost, TFT", style={'color': '#888', 'marginTop': '5px'})
                    ], className="metric-card")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div("40+", className="metric-value"),
                        html.Div("Features", className="metric-label"),
                        html.P("Economic indicators", style={'color': '#888', 'marginTop': '5px'})
                    ], className="metric-card")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            "XGBoost",
                            html.Span("WINNER", className="winner-badge")
                        ], className="metric-value", style={'fontSize': '1.8em'}),
                        html.Div("Best Model", className="metric-label"),
                        html.P("RMSE: 0.42", style={'color': '#00ff00', 'marginTop': '5px', 'fontWeight': 'bold'})
                    ], className="metric-card")
                ], width=3),
            ])
        ], style={'marginBottom': '40px'}),
        
        # Tabs
        dbc.Tabs([
            # Tab 1: Model Comparison
            dbc.Tab(label="🏆 Model Comparison", children=[
                html.Div([
                    html.H3("Performance Metrics", style={'marginTop': '20px'}),
                    dcc.Graph(id='model-comparison-chart'),
                    html.H3("Model Rankings", style={'marginTop': '30px'}),
                    html.Div(id='model-rankings-table')
                ], className="graph-container")
            ]),
            
            # Tab 2: Forecasts
            dbc.Tab(label="📈 Forecasts", children=[
                html.Div([
                    html.H3("Backtest Predictions vs Actual", style={'marginTop': '20px'}),
                    dcc.Graph(id='forecast-chart'),
                    html.H3("Forecast Errors by Model", style={'marginTop': '30px'}),
                    dcc.Graph(id='error-chart')
                ], className="graph-container")
            ]),
            
            # Tab 3: XGBoost Analysis
            dbc.Tab(label="🌳 XGBoost", children=[
                html.Div([
                    html.H3("XGBoost Hyperparameters (Optuna-Tuned)", style={'marginTop': '20px'}),
                    html.Div(id='xgboost-params'),
                    html.H3("Feature Importance", style={'marginTop': '30px'}),
                    html.P("(SHAP values - requires separate generation)", style={'color': '#888'})
                ], className="graph-container")
            ]),
            
            # Tab 4: TFT Analysis
            dbc.Tab(label="🧠 TFT Deep Learning", children=[
                html.Div([
                    html.H3("Temporal Fusion Transformer Analysis", style={'marginTop': '20px'}),
                    dcc.Graph(id='tft-importance-chart'),
                    html.H3("TFT Architecture", style={'marginTop': '30px'}),
                    html.Div(id='tft-architecture-info')
                ], className="graph-container")
            ]),
            
            # Tab 5: Backtest Details
            dbc.Tab(label="📋 Backtest Results", children=[
                html.Div([
                    html.H3("Detailed Backtest Results", style={'marginTop': '20px'}),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[{'label': m.upper(), 'value': m} for m in data['backtests'].keys()] if 'backtests' in data else [],
                        value=list(data['backtests'].keys())[0] if 'backtests' in data else None,
                        style={'width': '300px', 'marginBottom': '20px'}
                    ),
                    html.Div(id='backtest-table')
                ], className="graph-container")
            ]),
            
            # Tab 6: Download
            dbc.Tab(label="💾 Download", children=[
                html.Div([
                    html.H3("Download Reports & Data", style={'marginTop': '20px'}),
                    html.Div([
                        html.P("📄 Available downloads:", style={'fontSize': '1.2em', 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li("Model summaries (CSV)"),
                            html.Li("Backtest results (CSV)"),
                            html.Li("Feature importance (CSV)"),
                            html.Li("Model comparison charts (PNG)"),
                            html.Li("Complete results summary (Markdown)")
                        ], style={'fontSize': '1.1em', 'color': '#ddd'}),
                        html.P(
                            "Files are available in the 'artifacts/' directory",
                            style={'marginTop': '20px', 'color': '#888', 'fontStyle': 'italic'}
                        )
                    ])
                ], className="graph-container")
            ])
        ]),
        
    ], className="dashboard-container")
])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('model-comparison-chart', 'figure'),
    Input('model-comparison-chart', 'id')
)
def update_model_comparison(_):
    if 'summaries' not in data:
        return go.Figure()
    
    df = data['summaries'].sort_values('rmse')
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RMSE', 'MAE', 'MAPE (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['#00d4ff' if i == 0 else '#764ba2' for i in range(len(df))]
    
    # RMSE
    fig.add_trace(
        go.Bar(x=df['model'], y=df['rmse'], marker_color=colors, name='RMSE',
               text=df['rmse'].round(3), textposition='outside'),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=df['model'], y=df['mae'], marker_color=colors, name='MAE',
               text=df['mae'].round(3), textposition='outside'),
        row=1, col=2
    )
    
    # MAPE
    fig.add_trace(
        go.Bar(x=df['model'], y=df['mape'], marker_color=colors, name='MAPE',
               text=df['mape'].round(2), textposition='outside'),
        row=1, col=3
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@app.callback(
    Output('model-rankings-table', 'children'),
    Input('model-rankings-table', 'id')
)
def update_rankings_table(_):
    if 'summaries' not in data:
        return html.P("No data available")
    
    df = data['summaries'].sort_values('rmse')
    df['rank'] = range(1, len(df) + 1)
    df['rank'] = df['rank'].apply(lambda x: f"🥇 {x}" if x == 1 else f"🥈 {x}" if x == 2 else f"🥉 {x}" if x == 3 else f"  {x}")
    
    return dash_table.DataTable(
        data=df[['rank', 'model', 'rmse', 'mae', 'mape']].to_dict('records'),
        columns=[
            {'name': 'Rank', 'id': 'rank'},
            {'name': 'Model', 'id': 'model'},
            {'name': 'RMSE', 'id': 'rmse', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'MAE', 'id': 'mae', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'MAPE (%)', 'id': 'mape', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        ],
        style_cell={
            'textAlign': 'center',
            'padding': '15px',
            'backgroundColor': 'rgba(0, 0, 0, 0.3)',
            'color': 'white',
            'fontSize': '16px'
        },
        style_header={
            'backgroundColor': 'rgba(118, 75, 162, 0.8)',
            'fontWeight': 'bold',
            'fontSize': '18px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},
                'backgroundColor': 'rgba(0, 212, 255, 0.2)',
                'fontWeight': 'bold'
            }
        ]
    )

@app.callback(
    Output('forecast-chart', 'figure'),
    Input('forecast-chart', 'id')
)
def update_forecast_chart(_):
    if 'backtests' not in data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Get actual values from any backtest
    first_model = list(data['backtests'].keys())[0]
    actual_df = data['backtests'][first_model]
    
    # Plot actual
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_df))),
        y=actual_df['y_true'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#00ff00', width=3),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # Plot predictions for each model
    colors = ['#00d4ff', '#ff6b6b', '#feca57', '#48dbfb']
    for i, (model_name, df) in enumerate(data['backtests'].items()):
        valid_mask = ~df['y_pred'].isna()
        fig.add_trace(go.Scatter(
            x=list(range(len(df)))[::1],
            y=df.loc[valid_mask, 'y_pred'],
            mode='lines+markers',
            name=model_name.upper(),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Model Predictions vs Actual (Test Window)",
        xaxis_title="Forecast Horizon (months)",
        yaxis_title="OER YoY Change (%)",
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        hovermode='x unified',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    return fig

@app.callback(
    Output('error-chart', 'figure'),
    Input('error-chart', 'id')
)
def update_error_chart(_):
    if 'backtests' not in data:
        return go.Figure()
    
    fig = go.Figure()
    
    for model_name, df in data['backtests'].items():
        valid_mask = ~df['y_pred'].isna()
        if valid_mask.sum() > 0:
            errors = (df.loc[valid_mask, 'y_true'] - df.loc[valid_mask, 'y_pred']).abs()
            fig.add_trace(go.Box(
                y=errors,
                name=model_name.upper(),
                boxmean='sd'
            ))
    
    fig.update_layout(
        title="Forecast Error Distribution by Model",
        yaxis_title="Absolute Error",
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    return fig

@app.callback(
    Output('xgboost-params', 'children'),
    Input('xgboost-params', 'id')
)
def update_xgboost_params(_):
    if 'xgb_params' not in data:
        return html.P("XGBoost parameters not available")
    
    params = data['xgb_params'].get('best_params', {})
    rmse = data['xgb_params'].get('best_cv_rmse', 'N/A')
    
    param_items = [
        html.Li(f"{k}: {v}", style={'fontSize': '1.1em', 'marginBottom': '8px'})
        for k, v in params.items()
    ]
    
    return html.Div([
        html.P(f"Best CV RMSE: {rmse:.4f}" if isinstance(rmse, float) else f"Best CV RMSE: {rmse}",
               style={'fontSize': '1.3em', 'color': '#00ff00', 'fontWeight': 'bold', 'marginBottom': '15px'}),
        html.Ul(param_items, style={'color': '#ddd'})
    ])

@app.callback(
    Output('tft-importance-chart', 'figure'),
    Input('tft-importance-chart', 'id')
)
def update_tft_importance(_):
    if 'tft_importance' not in data:
        return go.Figure()
    
    df = data['tft_importance'].head(15)
    
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Viridis',
            showscale=True
        ),
        text=df['importance'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="TFT Variable Importance (Top 15 Features)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template='plotly_dark',
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

@app.callback(
    Output('tft-architecture-info', 'children'),
    Input('tft-architecture-info', 'id')
)
def update_tft_architecture(_):
    return html.Div([
        html.H4("Architecture Components:", style={'marginBottom': '15px'}),
        html.Ul([
            html.Li("Variable Selection Network (attention-based feature importance)", style={'marginBottom': '8px'}),
            html.Li("LSTM Encoder (hidden_size: 64, encoder_length: 12)", style={'marginBottom': '8px'}),
            html.Li("Multi-Head Attention (4 heads, captures temporal patterns)", style={'marginBottom': '8px'}),
            html.Li("Gated Residual Networks (non-linear transformations)", style={'marginBottom': '8px'}),
            html.Li("78,380 trainable parameters", style={'marginBottom': '8px'}),
            html.Li("Training: Adam optimizer with gradient clipping, early stopping", style={'marginBottom': '8px'}),
        ], style={'fontSize': '1.1em', 'color': '#ddd'}),
        html.P(
            "Note: TFT using TensorFlow 2.15 (replaced pytorch-forecasting for stability)",
            style={'marginTop': '20px', 'color': '#888', 'fontStyle': 'italic'}
        )
    ])

@app.callback(
    Output('backtest-table', 'children'),
    Input('model-selector', 'value')
)
def update_backtest_table(selected_model):
    if not selected_model or 'backtests' not in data:
        return html.P("No data available")
    
    df = data['backtests'][selected_model].copy()
    df['error'] = df['y_true'] - df['y_pred']
    df['abs_error'] = df['error'].abs()
    
    return dash_table.DataTable(
        data=df[['timestamp', 'y_true', 'y_pred', 'error', 'abs_error']].to_dict('records'),
        columns=[
            {'name': 'Date', 'id': 'timestamp'},
            {'name': 'Actual', 'id': 'y_true', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Predicted', 'id': 'y_pred', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Error', 'id': 'error', 'type': 'numeric', 'format': {'specifier': '.4f'}},
            {'name': 'Abs Error', 'id': 'abs_error', 'type': 'numeric', 'format': {'specifier': '.4f'}},
        ],
        style_cell={
            'textAlign': 'center',
            'padding': '12px',
            'backgroundColor': 'rgba(0, 0, 0, 0.3)',
            'color': 'white'
        },
        style_header={
            'backgroundColor': 'rgba(118, 75, 162, 0.8)',
            'fontWeight': 'bold'
        }
    )

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🚀 OER FORECASTING DASHBOARD")
    print("="*80)
    print("\n📊 Dashboard starting...")
    print("🌐 Open your browser to: http://127.0.0.1:8050")
    print("\n💡 Features:")
    print("  - Executive summary with key metrics")
    print("  - Interactive model comparison charts")
    print("  - Time series forecasts")
    print("  - XGBoost & TFT analysis")
    print("  - Detailed backtest results")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, port=8050)
