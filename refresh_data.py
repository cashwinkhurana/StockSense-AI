
import yfinance as yf
from pathlib import Path
import subprocess
import sys

stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

print("\n--- StockSense AI: Refreshing Market Data ---")

for stock in stocks:
    print(f"Downloading {stock}...")

    df = yf.download(
        stock,
        start="2015-01-01",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"WARNING: No data received for {stock}")
        sys.exit(1)

    df.to_csv(data_folder / f"{stock}.csv")

    print(f"{stock}: {len(df)} trading days downloaded")

# Rebuild processed datasets
print("\nPreparing data...")
subprocess.run(
    [sys.executable, "prepare_data.py"],
    check=True
)

# Recalculate technical indicators
print("\nUpdating technical indicators...")
subprocess.run(
    [sys.executable, "feature_engineering.py"],
    check=True
)


# Retrain models and generate updated predictions
print("\nUpdating prediction models...")

subprocess.run(
    [sys.executable, "prediction_engine.py"],
    check=True
)

print("\nStockSense AI update completed!")
print("Market data, features and models are updated.")
