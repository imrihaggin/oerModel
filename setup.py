from setuptools import find_packages, setup


setup(
    name="oer-model",
    version="0.2.0",
    description="Modular forecasting platform for Owners' Equivalent Rent.",
    author="imrihaggin",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1",
        "numpy>=1.23",
        "pandas>=1.5",
        "pandas-datareader>=0.10",
        "pyyaml>=6.0",
        "scikit-learn>=1.2",
    "statsmodels>=0.13",
        "plotly>=5.15",
        "dash>=2.11",
        "dash-bootstrap-components>=1.4",
        "joblib>=1.2",
    ],
    extras_require={
        "bloomberg": ["xbbg>=0.7.7"],
        "xgboost": ["xgboost>=1.7"],
        "deep": ["pytorch-lightning>=2.0", "pytorch-forecasting>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "oer-model=oer_model.cli:cli",
        ]
    },
)