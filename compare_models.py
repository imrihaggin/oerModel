"""Compare all models from backtest results."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read all summary files
summaries = []
for summary_file in Path('artifacts').glob('summary_*.csv'):
    df = pd.read_csv(summary_file)
    summaries.append(df)

results = pd.concat(summaries, ignore_index=True)
results = results.sort_values('rmse')

print("="*80)
print("MODEL COMPARISON - OER FORECASTING")
print("="*80)
print(results.to_string(index=False))
print("\n" + "="*80)
print(f"BEST MODEL: {results.iloc[0]['model'].upper()}")
print(f"  RMSE: {results.iloc[0]['rmse']:.4f}")
print(f"  MAE: {results.iloc[0]['mae']:.4f}")
print(f"  MAPE: {results.iloc[0]['mape']:.2f}%")
print("="*80)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RMSE comparison
ax1 = axes[0]
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results))]
ax1.barh(results['model'], results['rmse'], color=colors)
ax1.set_xlabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(results['rmse']):
    ax1.text(v + 0.05, i, f'{v:.3f}', va='center', fontsize=10)

# MAE comparison
ax2 = axes[1]
ax2.barh(results['model'], results['mae'], color=colors)
ax2.set_xlabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(results['mae']):
    ax2.text(v + 0.05, i, f'{v:.3f}', va='center', fontsize=10)

# MAPE comparison
ax3 = axes[2]
ax3.barh(results['model'], results['mape'], color=colors)
ax3.set_xlabel('MAPE (%)', fontsize=12, fontweight='bold')
ax3.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, v in enumerate(results['mape']):
    ax3.text(v + 1, i, f'{v:.2f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('artifacts/model_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to artifacts/model_comparison.png")

# Create backtest predictions visualization
fig, ax = plt.subplots(figsize=(14, 6))

for model_file in Path('artifacts').glob('backtest_*.csv'):
    model_name = model_file.stem.replace('backtest_', '')
    df = pd.read_csv(model_file)
    df = df[~df['y_pred'].isna()]  # Filter NaN predictions
    
    if len(df) > 0:
        ax.plot(df.index, df['y_pred'], marker='o', label=f'{model_name}', linewidth=2)

# Plot actual values
lasso_df = pd.read_csv('artifacts/backtest_lasso.csv')
ax.plot(lasso_df.index, lasso_df['y_true'], marker='s', label='Actual', 
        color='black', linewidth=2.5, markersize=8, linestyle='--')

ax.set_xlabel('Forecast Horizon (months ahead)', fontsize=12, fontweight='bold')
ax.set_ylabel('OER YoY Change (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Predictions vs Actual (Test Window: 2024-04 to 2025-07)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/backtest_predictions.png', dpi=300, bbox_inches='tight')
print("Backtest predictions plot saved to artifacts/backtest_predictions.png\n")
