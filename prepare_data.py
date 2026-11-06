import pandas as pd
from pathlib import Path

data_folder = Path("data")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

for stock in stocks:
    # Read downloaded Yahoo Finance data
    df = pd.read_csv(
        data_folder / f"{stock}.csv",
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    # Flatten column names
    df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"

    # Remove duplicate dates and sort chronologically
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Convert prices and volume to numeric values
    columns = ["Open", "High", "Low", "Close", "Volume"]
    df[columns] = df[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove incomplete records
    df = df.dropna(subset=columns)

    # Create our prediction target:
    # Percentage return over the next 5 trading days
    df["Target_5D_Return"] = (
        df["Close"].shift(-5) / df["Close"] - 1
    ) * 100

    # Save cleaned dataset
    df.to_csv(output_folder / f"{stock}_processed.csv")

    print(f"{stock}: {len(df)} rows processed")

print("Data preparation completed!")