import pandas as pd

df = pd.read_csv('data/raw/raw_combined.csv')
print(f"Raw data shape: {df.shape}")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"\nFirst valid date for each series:")
print("-" * 60)

for col in df.columns:
    if col != 'date':
        first_valid_idx = df[col].first_valid_index()
        if first_valid_idx is not None:
            first_date = df.loc[first_valid_idx, 'date']
            count = df[col].notna().sum()
            print(f"{col:30s}: {first_date:12s} ({count:3d} months)")
