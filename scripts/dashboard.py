#!/usr/bin/env python
"""Launch the OER forecasting dashboard."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oer_model.viz.dashboard import create_dashboard


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / "artifacts"
    forecasts_path = artifacts_dir / "latest_forecasts.csv"
    
    # Check if required files exist
    if not artifacts_dir.exists():
        print(f"❌ Error: Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)
    
    if not forecasts_path.exists():
        print(f"❌ Error: Forecasts file not found: {forecasts_path}")
        print("💡 Run: python -m oer_model.cli forecast --horizon 12")
        sys.exit(1)
    
    app = create_dashboard(artifacts_dir, forecasts_path)
    print("\n" + "="*60)
    print("🚀 OER Forecasting Dashboard")
    print("="*60)
    print("\n📊 Dashboard running at: http://127.0.0.1:8050")
    print("\n📁 Loading data from:")
    print(f"   - {artifacts_dir}/backtest_*.csv")
    print(f"   - {forecasts_path}")
    print("\n⚡ Press CTRL+C to stop\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
