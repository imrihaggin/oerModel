"""Launch the Plotly Dash dashboard."""

from pathlib import Path

from oer_model.config import load_config
from oer_model.viz.dashboard import create_dashboard


def main(port: int = 8050) -> None:
    config = load_config()
    artifacts_dir = config.artifacts_dir
    forecasts_path = artifacts_dir / "latest_forecasts.csv"
    app = create_dashboard(artifacts_dir, forecasts_path)
    app.run_server(debug=True, port=port)


if __name__ == "__main__":
    main()
