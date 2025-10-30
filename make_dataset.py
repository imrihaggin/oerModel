"""
Bloomberg BQL Data Fetching Script for OER Forecasting

This script demonstrates the correct usage of Bloomberg's BQL API (Beta)
for fetching economic time series data for the BQNT terminal environment.

BQL is fundamentally different from traditional Bloomberg APIs:
- Uses a functional, data-frame oriented syntax
- Returns data in a structured pandas DataFrame format
- Requires proper handling of the response structure

Usage:
    python make_dataset.py data/raw/bloomberg_data.csv
"""

import bql
import pandas as pd
import click
from pathlib import Path
from datetime import datetime

# --- BLOOMBERG TICKER DEFINITIONS ---
# These are Bloomberg terminal tickers for the key OER predictors
BLOOMBERG_TICKERS = {
    # Target Variable
    'oer_cpi_sa': 'CPIQOEPS Index',  # OER CPI Seasonally Adjusted (to be converted to YoY)
    
    # Primary Housing Predictors (with known lead times from research)
    'zori_yoy': 'ZRIOAYOY Index',     # Zillow Observed Rent Index YoY (12-month lead)
    'zhvi_sa': 'ZHVIACUR Index',      # Zillow Home Value Index SA (16-month lead, needs YoY conversion)
    'case_shiller_sa': 'SPCS20RSA Index',  # S&P Case-Shiller 20-City Home Price Index SA
    
    # Labor Market Indicators
    'unemployment_rate': 'USURTOT Index',   # Unemployment Rate Total SA
    'ahe_total': 'AHE TOTL Index',          # Average Hourly Earnings Total Private SA
    'eci_total_comp': 'ECICCVYY Index',     # Employment Cost Index Total Comp YoY
    
    # Additional Housing Market Indicators
    'housing_starts': 'NHSPSTOT Index',     # New Housing Starts Total SA
    'new_home_sales': 'NHSLTOT Index',      # New Home Sales Total SA
}

# Lead times based on empirical research (months)
LEAD_TIMES = {
    'zori_yoy': 12,
    'zhvi_yoy': 16,
    'case_shiller_yoy': 16,
}

def fetch_data_from_bloomberg(tickers: list, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Fetches historical time series data using Bloomberg BQL API.
    
    BQL-specific notes:
    - BQL returns data in a specific nested structure
    - The response object contains multiple DataFrames
    - Must properly unpack the response to get time series data
    
    Args:
        tickers: List of Bloomberg ticker strings (e.g., ['CPIQOEPS Index'])
        start_date: Start date as string 'YYYY-MM-DD'
        end_date: End date as string 'YYYY-MM-DD' or None for today
        
    Returns:
        DataFrame with dates as index and ticker data as columns
    """
    try:
        # Initialize BQL service
        bq = bql.Service()
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date or 'today'}...")
        
        # Construct BQL request - this is the correct syntax for time series
        # bq.data.px_last() gets the last price/value
        # dates parameter accepts a range function for historical data
        if end_date is None or end_date.lower() == 'today':
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        request = bq.Request(
            tickers,
            {
                'value': bq.data.px_last(dates=bq.func.range(start_date, end_date), frq='M')
            }
        )
        
        # Execute request
        response = bq.execute(request)
        
        # BQL response structure:
        # response is a list where response[0] contains the data result
        # response[0].df() converts to pandas DataFrame
        # The DataFrame has MultiIndex columns: (ticker, field)
        df_raw = response[0].df()
        
        # The raw response has a specific structure - let's reshape it
        # Typically: index=DATE, columns=MultiIndex with (ID, field_name)
        if isinstance(df_raw.columns, pd.MultiIndex):
            # Flatten MultiIndex columns, keeping just the ticker ID
            df_raw.columns = [col[0] for col in df_raw.columns]
        
        # Ensure monthly frequency
        df_raw = df_raw.asfreq('MS')  # Month start frequency
        
        print(f"Successfully fetched {len(df_raw)} rows")
        return df_raw
        
    except ImportError:
        raise ImportError(
            "BQL library not found. This script requires Bloomberg's BQL library, "
            "which is only available in the Bloomberg terminal environment (BQNT)."
        )
    except Exception as e:
        print(f"Error during BQL request: {e}")
        print("Note: BQL is a beta API and may have changed. Check Bloomberg documentation.")
        raise


def process_raw_data(df_raw: pd.DataFrame, ticker_map: dict) -> pd.DataFrame:
    """
    Process raw Bloomberg data into clean feature matrix.
    
    Args:
        df_raw: Raw DataFrame from Bloomberg with tickers as columns
        ticker_map: Dictionary mapping friendly names to Bloomberg tickers
        
    Returns:
        Processed DataFrame with renamed columns and forward-filled values
    """
    # Create reverse mapping for renaming
    reverse_map = {v: k for k, v in ticker_map.items()}
    
    # Rename columns to friendly names
    df_processed = df_raw.rename(columns=reverse_map)
    
    # Forward fill missing values (common in economic time series)
    df_processed = df_processed.ffill()
    
    # Sort by date index
    df_processed = df_processed.sort_index()
    
    return df_processed


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features following the research methodology.
    
    Key transformations based on OER forecasting literature:
    1. Convert level indices to YoY percentage changes
    2. Apply empirically-determined lags to leading indicators
    3. Create the target variable (OER YoY growth)
    
    Args:
        df: Processed DataFrame with cleaned Bloomberg data
        
    Returns:
        DataFrame with engineered features ready for modeling
    """
    df_features = df.copy()
    
    # --- 1. Calculate Year-over-Year percentage changes ---
    # OER (target variable)
    if 'oer_cpi_sa' in df_features.columns:
        df_features['oer_yoy'] = df_features['oer_cpi_sa'].pct_change(12) * 100
    
    # ZHVI (needs YoY conversion before lagging)
    if 'zhvi_sa' in df_features.columns:
        df_features['zhvi_yoy'] = df_features['zhvi_sa'].pct_change(12) * 100
        
    # Case-Shiller (also needs YoY conversion)
    if 'case_shiller_sa' in df_features.columns:
        df_features['case_shiller_yoy'] = df_features['case_shiller_sa'].pct_change(12) * 100
    
    # Average Hourly Earnings
    if 'ahe_total' in df_features.columns:
        df_features['ahe_yoy'] = df_features['ahe_total'].pct_change(12) * 100
    
    # --- 2. Apply Lead-Time Lags to Predictors ---
    # CRITICAL: This respects the structural lag between market indicators and OER
    for feature, lag_months in LEAD_TIMES.items():
        if feature in df_features.columns:
            df_features[f'{feature}_lag{lag_months}m'] = df_features[feature].shift(lag_months)
    
    # --- 3. Additional Lags for All Features (for model flexibility) ---
    # Include 3, 6, 9 month lags for labor market variables
    lag_features = ['unemployment_rate', 'ahe_yoy', 'eci_total_comp']
    for feature in lag_features:
        if feature in df_features.columns:
            for lag in [3, 6, 9]:
                df_features[f'{feature}_lag{lag}m'] = df_features[feature].shift(lag)
    
    # --- 4. Rolling averages (smooth out volatility) ---
    if 'unemployment_rate' in df_features.columns:
        df_features['unemployment_rate_ma3'] = df_features['unemployment_rate'].rolling(3).mean()
        df_features['unemployment_rate_ma6'] = df_features['unemployment_rate'].rolling(6).mean()
    
    return df_features


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.option('--start-date', default='2010-01-01', help='Start date for data fetch (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for data fetch (YYYY-MM-DD), defaults to today')
def main(output_filepath: str, start_date: str, end_date: str):
    """
    Fetch Bloomberg data via BQL and prepare features for OER forecasting.
    
    This is a standalone script for Bloomberg Terminal (BQNT) environments.
    For production use, integrate with src/oer_model/data/bloomberg.py module.
    
    Example:
        python make_dataset.py data/raw/bloomberg_data.csv --start-date 2010-01-01
    """
    print("=" * 70)
    print("OER FORECASTING - BLOOMBERG BQL DATA PIPELINE")
    print("=" * 70)
    
    # --- 1. Fetch Data from Bloomberg ---
    ticker_list = list(BLOOMBERG_TICKERS.values())
    
    try:
        df_raw = fetch_data_from_bloomberg(ticker_list, start_date, end_date)
    except ImportError as e:
        print(f"\n{e}")
        print("\nNOTE: This script is designed for Bloomberg Terminal (BQNT) environment.")
        print("For development without Bloomberg access, use the FRED/Zillow data pipeline instead.")
        return
    
    if df_raw is None or df_raw.empty:
        print("ERROR: Failed to fetch data from Bloomberg")
        return
    
    # --- 2. Process Raw Data ---
    print("\nProcessing raw data...")
    df_processed = process_raw_data(df_raw, BLOOMBERG_TICKERS)
    print(f"Processed {len(df_processed)} months of data")
    print(f"Columns: {list(df_processed.columns)}")
    
    # --- 3. Engineer Features ---
    print("\nEngineering features...")
    df_features = engineer_features(df_processed)
    print(f"Created {len(df_features.columns)} total features")
    
    # --- 4. Prepare Final Dataset ---
    # Move target to first column
    if 'oer_yoy' in df_features.columns:
        target_col = 'oer_yoy'
        feature_cols = [col for col in df_features.columns if col != target_col]
        df_final = df_features[[target_col] + feature_cols]
    else:
        df_final = df_features
    
    # --- 5. Save Output ---
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_csv(output_path)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Rows: {len(df_final)}")
    print(f"Columns: {len(df_final.columns)}")
    print(f"Date Range: {df_final.index[0]} to {df_final.index[-1]}")
    print(f"Output: {output_path.absolute()}")
    print(f"\nTarget variable: oer_yoy (OER Year-over-Year % change)")
    print(f"Missing values: {df_final.isnull().sum().sum()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
