import yfinance as yf
from pathlib import Path

# US stocks we want to analyse
stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

# Create a folder to store our datasets
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Download historical stock prices
for stock in stocks:
    print(f"Downloading {stock}...")

    data = yf.download(
        stock,
        start="2015-01-01",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        print(f"No data found for {stock}")
        continue

    # Save each stock's data as a CSV file
    data.to_csv(data_folder / f"{stock}.csv")

    print(f"Saved {stock}: {len(data)} trading days")

print("All downloads completed!")
