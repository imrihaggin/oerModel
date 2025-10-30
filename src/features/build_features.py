# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd

def main():
    """
    Reads the raw data, creates features, and saves the processed dataset.
    """
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / 'data' / 'raw' / 'raw_market_data.csv'
    processed_data_path = project_dir / 'data' / 'processed' / 'OER_Predictors_FINAL.csv'

    if not raw_data_path.exists():
        print(f"Raw data not found at {raw_data_path}.")
        print("Please run `python src/data/make_dataset.py` first.")
        return

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path, index_col='Date', parse_dates=True)

    print("Building features...")
    # Ensure columns exist before processing
    if 'CPI O E Index' in df.columns:
        df['CPI O E Index_yoy'] = df['CPI O E Index'].pct_change(12) * 100
    
    if 'ZRI O A YOY Index' in df.columns:
        # Create lags for Zillow Observed Rent Index YoY
        df['ZRIOAYOY Index_lagged'] = df['ZRI O A YOY Index'].shift(12)

    if 'ZHVIACUR' in df.columns:
        # Create Year-over-Year for Zillow Home Value Index
        df['ZHVIACUR Index_yoy'] = df['ZHVIACUR'].pct_change(12) * 100
        # Create lags for ZHVI YoY
        df['ZHVIACUR Index_yoy_lagged'] = df['ZHVIACUR Index_yoy'].shift(15)

    # Define the target variable and final features
    final_cols = [
        'CPI O E Index_yoy',
        'ZRIOAYOY Index_lagged',
        'ZHVIACUR Index_yoy_lagged'
    ]
    
    # Filter for columns that actually exist in the dataframe
    existing_cols = [col for col in final_cols if col in df.columns]
    
    final_df = df[existing_cols].copy()

    # Drop rows with NaN values resulting from lags and pct_change
    final_df.dropna(inplace=True)

    # --- Save Processed Data ---
    final_df.to_csv(processed_data_path)
    print(f"Successfully saved processed data to {processed_data_path}")


if __name__ == '__main__':
    main()