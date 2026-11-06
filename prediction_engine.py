
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier

stocks = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

features = [
    "Daily_Return",
    "Volatility_20",
    "RSI_14",
    "BB_Width",
    "Volume_Change",
    "Momentum_5D",
    "MA10_Ratio",
    "MA50_Ratio"
]

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

print("\n--- StockSense AI Predictions ---")

for stock in stocks:
    df = pd.read_csv(
        f"data/features/{stock}_features.csv",
        parse_dates=["Date"]
    )

    df = df.sort_values("Date")

    # Use only rows with known future outcomes for training
    train_df = df.dropna(
        subset=features + ["Target_5D_Return"]
    )

    X_train = train_df[features]
    y_train = (
        train_df["Target_5D_Return"] > 0
    ).astype(int)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Latest available market observation
    latest = df.dropna(subset=features).iloc[[-1]]

    probability_up = model.predict_proba(
        latest[features]
    )[0, 1]

    prediction = (
        "UP" if probability_up >= 0.5 else "DOWN/FLAT"
    )

    # Save model
    joblib.dump(
        model,
        models_folder / f"{stock}_final.pkl"
    )

    print(
        f"{stock}: {prediction} | "
        f"UP Probability: {probability_up:.2%}"
    )

print("\nPredictions generated successfully!")
