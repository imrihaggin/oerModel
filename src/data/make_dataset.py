# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import pandas_datareader.data as web
import datetime as dt

# (Assuming blpapi is installed and configured)
# If you don't have a Bloomberg terminal, you'll need to use
# another data provider for the Bloomberg data.
try:
    import blpapi
    from xbbg import blp
    BBG_AVAILABLE = True
except ImportError:
    BBG_AVAILABLE = False
    print("Bloomberg API not found. Skipping Bloomberg data download.")


# Define tickers
BBG_TICKERS = ['CPI O E Index', 'ZRI O A YOY Index']
FRED_TICKERS = ['CPIUFSL', 'ZHVIACUR']

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    project_dir = Path(__file__).resolve().parents[2]
    output_path = project_dir / 'data' / 'raw' / 'raw_market_data.csv'

    # --- Fetch Bloomberg Data ---
    if BBG_AVAILABLE:
        print("Fetching Bloomberg data...")
        bbg_data = blp.bdh(
            BBG_TICKERS,
            'PX_LAST',
            '1/1/2014',
            end_date=dt.date.today().strftime('%Y%m%d')
        )
        # Flatten multi-level column names
        bbg_data.columns = [col[0] for col in bbg_data.columns]
    else:
        print("Creating placeholder for Bloomberg data.")
        # Create an empty DataFrame with expected columns if BBG is not available
        bbg_data = pd.DataFrame(columns=BBG_TICKERS)


    # --- Fetch FRED Data ---
    print("Fetching FRED data...")
    start = dt.datetime(2014, 1, 1)
    end = dt.datetime.today()
    fred_data = web.DataReader(FRED_TICKERS, 'fred', start, end)

    # --- Combine Data ---
    print("Combining data sources...")
    # Resample FRED data to month-end to align with Bloomberg's typical reporting
    fred_data_resampled = fred_data.resample('M').last()
    
    # Combine the datasets
    combined_data = pd.merge(bbg_data, fred_data_resampled, left_index=True, right_index=True, how='outer')
    combined_data.index.name = 'Date'

    # Forward fill to handle non-trading days or data alignment issues
    combined_data = combined_data.ffill()

    # --- Save Raw Data ---
    combined_data.to_csv(output_path)
    print(f"Successfully saved raw combined data to {output_path}")


if __name__ == '__main__':
    main()