# 📈 StockSense AI

### Machine Learning-Powered Stock Market Prediction & Analytics

StockSense AI is an end-to-end machine learning project designed to analyse historical US stock market data and generate experimental predictions of stock price direction over the next five trading days.

The project combines financial data engineering, technical analysis, XGBoost classification, walk-forward validation and an interactive Streamlit dashboard.

## Key Features

- **Stock Market Analysis:** Supports AAPL, MSFT, NVDA, TSLA and GOOGL.
- **Machine Learning Predictions:** Uses XGBoost to classify potential upward and downward price movements.
- **Technical Indicators:** RSI, MACD, Bollinger Bands, moving averages, volatility and momentum.
- **Interactive Dashboard:** Candlestick charts, model scores and historical performance comparisons.
- **Market Data Refresh:** Downloads updated historical prices and regenerates model predictions.
- **Model Evaluation:** Compares prediction accuracy against an Always-UP baseline using time-series cross-validation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Collection | yfinance |
| Data Processing | pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Technical Analysis | ta |
| Visualisation | Plotly |
| Web Application | Streamlit |
| Model Storage | joblib |

## Machine Learning Methodology

The model predicts whether a stock's closing price will increase over the next five trading days.

The pipeline consists of:

1. Downloading historical US stock market data.
2. Cleaning and preparing financial datasets.
3. Calculating technical indicators.
4. Training regularised XGBoost classification models.
5. Evaluating performance using five-fold time-series cross-validation with a five-observation gap.
6. Generating predictions from the latest available market data.

## Historical Model Performance

Preliminary walk-forward evaluation:

| Stock | Model Accuracy | Always-UP Baseline |
|---|---:|---:|
| AAPL | 56.05% | 58.56% |
| MSFT | 56.17% | 59.05% |
| NVDA | 58.07% | 58.77% |
| TSLA | 50.53% | 53.17% |
| GOOGL | 55.31% | 57.12% |

The model did not outperform the baseline in these experiments. These findings demonstrate the challenges of financial forecasting and the importance of evaluating models against simple benchmarks.

Further validation, probability calibration and independent testing are required before interpreting predictions as reliable forecasts.

## Running Locally

Clone the repository and navigate to the project directory.

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the datasets and prediction models:

```bash
python refresh_data.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Project Objectives

StockSense AI was developed as a practical demonstration of:

- Financial data engineering and automation
- Feature engineering and technical analysis
- Machine learning model development
- Time-series evaluation and baseline comparisons
- Interactive analytics dashboard development
- End-to-end Python application integration

## Disclaimer

StockSense AI is an educational and experimental machine learning project. Its predictions are not calibrated probabilities, and historical testing has not demonstrated superiority over simple baseline strategies.

The application is not intended to provide investment or financial advice.