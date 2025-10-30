# src/data/make_dataset.py

"""
This script fetches raw data from the Bloomberg API, processes it,
engineers features, and saves the final model-ready dataset.
"""

import bql
import pandas as pd
import os
import click # A library for creating command-line interfaces

# --- 1. Define Constants and Tickers ---
SERIES_TICKERS = {
'CPIQOEPS': 'CPIQOEPS Index', # Seasonally adjusted not YoY; TARGET VAR to be YoY
# 'CPRHOERY': 'CPRHOERY Index', # OER NSA YoY
'ZRIOAYOY': 'ZRIOAYOY Index', # All homes rent YoY smoothed seasonally adjusted
'ZHVIACUR': 'ZHVIACUR Index', # All homes value smoothed seasonally adjusted to be YoY
'UNRATE': 'USURTOT Index', # Unemployment rate total seasonally adjusted
'AHE': 'AHE TOTL Index',  # US Average hour earnings seasonally adjusted
'ECICCVYY': 'ECICCVYY Index', # YOY SA
}
ZORI_LEAD_TIME = 12
ZHVI_LEAD_TIME = 16
START_DATE = '2010-01-01'
END_DATE = 'today'

def fetch_data_from_bloomberg(tickers, start_date, end_date):
    """Fetches historical time series data for a list of tickers using BQL."""
    bq = bql.Service()
    print(f"Fetching data for {len(tickers)} series from {start_date} to {end_date}...")

    bql_request = bql.Request(
        tickers, 
        {'price': bq.data.px_last(dates=bq.func.range(start_date, end_date))}
    )
    try:
        response = bq.execute(bql_request)
    except Exception as e:
        print(f"Error executing BQL request: {e}")
        return None

    print("Data fetching complete.")
    return response[0].df()
    
@click.command()
@click.argument('output_filepath', type=click.Path())
    
def main(output_filepath):
    """ Main function to run the data processing pipeline."""
    print("--- Starting data preparation pipeline ---")
    
    # --- 2. Execute Data Pull and Initial Cleaning ---
    ticker_list = list(SERIES_TICKERS.values())
    df_raw = fetch_data_from_bloomberg(ticker_list, START_DATE, END_DATE)
    
    df_raw = df_raw.ffill()
    
    reverse_ticker_map = {v: k for k, v in SERIES_TICKERS.items()}
    df_combined = df_raw.rename(columns=reverse_ticker_map)

    df_features = df_combined.pivot_table(
        index='DATE',
        columns=df_combined.index,
        values='price'
    )
    
    # --- 3. Transform Data to Year-over-Year Growth Rates ---
    # df_yoy = pd.DataFrame(index=df_combined.index)
    for col in ['CPIQOEPS Index', 'ZHVIACUR Index']:
        df_features[f'{col}_yoy'] = df_features[col].pct_change(12) * 100
        
    # --- 4. Feature Engineering: Create Lagged Predictors ---
    df_features['ZRIOAYOY Index_lagged'] = df_features['ZRIOAYOY Index'].shift(ZORI_LEAD_TIME)
    df_features['ZHVIACUR Index_yoy_lagged'] = df_features['ZHVIACUR Index_yoy'].shift(ZHVI_LEAD_TIME)
    
    for col in df_features:
        for lag in [30, 60, 90]:
            df_features[f'{col}_lag{lag}d'] = df_features[col].shift(lag)
    
    # --- 5. Final Processing for Modeling ---
    print(df_features.columns)
    print(df_features)
    
    target_col = 'CPIQOEPS Index_yoy'
    feature_cols = [col for col in df_features.columns if col!= target_col]
    
    df_model_ready = df_features[[target_col] + feature_cols]
    # df_model_ready.dropna(inplace=True)
    
    # --- 6. Save Processed Data ---
    # The output path is now an argument passed to the script
    df_model_ready.to_csv(output_filepath)
    print(f"\nFinal model-ready data has {df_model_ready.shape} rows and {df_model_ready.shape[1]} columns.")
    print(f"Successfully saved to: {output_filepath}")
    print("--- Data preparation pipeline finished ---")

if __name__ == '__main__':
    # This allows the script to be run from the command line
    main()
