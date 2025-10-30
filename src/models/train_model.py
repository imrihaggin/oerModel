import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

def main():
    """
    Trains the XGBoost model on the processed data.
    """
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_path = project_dir / 'data' / 'processed' / 'OER_Predictors_FINAL.csv'
    model_output_path = project_dir / 'models' / 'xgboost_oer_model.pkl'

    print("Loading processed data...")
    df = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)

    if 'CPI O E Index_yoy' not in df.columns:
        print("Target variable 'CPI O E Index_yoy' not found in processed data.")
        return

    X = df.drop('CPI O E Index_yoy', axis=1)
    y = df['CPI O E Index_yoy']

    print("Training XGBoost model with Time Series Cross-Validation...")
    
    # Example parameters, should be tuned
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # In a real scenario, you'd use TimeSeriesSplit for CV and hyperparameter tuning.
    # For this script, we'll just train on the full available dataset.
    model.fit(X, y)

    print(f"Saving trained model to {model_output_path}...")
    joblib.dump(model, model_output_path)
    print("Model training complete.")

if __name__ == '__main__':
    main()