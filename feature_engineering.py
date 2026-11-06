
import pandas as pd
import numpy as np
from pathlib import Path

from ta.trend import MACD
from ta.volatility import BollingerBands

# Folder configuration
input_folder = Path("data/processed")
output_folder = Path("data/features")
output_folder.mkdir(parents=True, exist_ok=True)

# US stocks
stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

for stock in stocks:
    print(f"Processing {stock}...")

    # Load historical stock data
    df = pd.read_csv(
        input_folder / f"{stock}_processed.csv",
        parse_dates=["Date"]
    )

    df = df.sort_values("Date").copy()

    # -----------------------------------
    # 1. DAILY RETURNS
    # -----------------------------------
    df["Daily_Return"] = (
        df["Close"].pct_change() * 100
    )

    # -----------------------------------
    # 2. MOVING AVERAGES
    # -----------------------------------
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # -----------------------------------
    # 3. VOLATILITY (20 DAYS)
    # -----------------------------------
    df["Volatility_20"] = (
        df["Daily_Return"].rolling(20).std()
        * np.sqrt(252)
    )

    # -----------------------------------
    # 4. RELATIVE STRENGTH INDEX (RSI)
    # -----------------------------------
    delta = df["Close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = losses.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI_14"] = 100 - (100 / (1 + rs))

    df.loc[avg_loss == 0, "RSI_14"] = 100

    df.loc[
        (avg_gain == 0) & (avg_loss == 0),
        "RSI_14"
    ] = 50

    # -----------------------------------
    # 5. MACD
    # -----------------------------------
    macd = MACD(close=df["Close"])

    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    # -----------------------------------
    # 6. BOLLINGER BANDS
    # -----------------------------------
    bb = BollingerBands(
        close=df["Close"],
        window=20
    )

    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()

    # -----------------------------------
    # 7. VOLUME CHANGE
    # -----------------------------------
    df["Volume_Change"] = (
        df["Volume"].pct_change() * 100
    )

    # -----------------------------------
    # 8. FIVE-DAY MOMENTUM
    # -----------------------------------
    df["Momentum_5D"] = (
        df["Close"].pct_change(5) * 100
    )

    # -----------------------------------
    # 9. MOVING AVERAGE RATIOS
    # -----------------------------------
    df["MA10_Ratio"] = df["Close"] / df["MA_10"]
    df["MA50_Ratio"] = df["Close"] / df["MA_50"]

    # -----------------------------------
    # 10. CLEAN FEATURE DATA
    # -----------------------------------
    feature_columns = [
        "Daily_Return",
        "MA_10",
        "MA_50",
        "Volatility_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Diff",
        "BB_High",
        "BB_Low",
        "BB_Width",
        "Volume_Change",
        "Momentum_5D",
        "MA10_Ratio",
        "MA50_Ratio"
    ]

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows with incomplete features
    df = df.dropna(subset=feature_columns)

    # Keep rows with unknown future targets
    # for later live predictions

    # -----------------------------------
    # 11. SAVE DATASET
    # -----------------------------------
    output_file = (
        output_folder / f"{stock}_features.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"{stock}: {len(df)} rows prepared")

print("\nFeature engineering completed!")
