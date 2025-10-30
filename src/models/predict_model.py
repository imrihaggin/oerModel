import pandas as pd
import joblib
from pathlib import Path

def generate_forecast(model, latest_features):
    """
    Generates a forecast using the trained model.
    """
    # The model expects a DataFrame, so we create one from the latest_features
    forecast = model.predict(latest_features)
    return forecast

def main():
    """
    Loads a trained model and makes a prediction.
    """
    project_dir = Path(__file__).resolve().parents[2]
    model_path = project_dir / 'models' / 'xgboost_oer_model.pkl'
    processed_data_path = project_dir / 'data' / 'processed' / 'OER_Predictors_FINAL.csv'

    if not model_path.exists():
        print(f"Model not found at {model_path}.")
        print("Please run `python src/models/train_model.py` first.")
        return
        
    if not processed_data_path.exists():
        print(f"Processed data not found at {processed_data_path}.")
        return

    print("Loading trained model...")
    model = joblib.load(model_path)
    
    # For demonstration, we'll use the last available data point to "predict"
    df = pd.read_csv(processed_data_path, index_col='Date', parse_dates=True)
    
    if 'CPI O E Index_yoy' not in df.columns:
        print("Target variable not found in data.")
        return

    X = df.drop('CPI O E Index_yoy', axis=1)
    
    if X.empty:
        print("No data available for prediction.")
        return

    latest_features = X.iloc[[-1]] # Get the last row as a DataFrame
    
    print(f"Generating forecast based on features from {latest_features.index[0].date()}:")
    print(latest_features)
    
    forecast = generate_forecast(model, latest_features)
    
    print(f"\nPredicted 'CPI O E Index_yoy': {forecast[0]:.4f}")


if __name__ == '__main__':
    main()
