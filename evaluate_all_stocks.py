
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

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

results = []

for stock in stocks:
    df = pd.read_csv(
        f"data/features/{stock}_features.csv",
        parse_dates=["Date"]
    )

    df = df.sort_values("Date")
    df = df.dropna(
        subset=features + ["Target_5D_Return"]
    )

    X = df[features]
    y = (df["Target_5D_Return"] > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=5, gap=5)

    model_scores = []
    baseline_scores = []

    for train_idx, test_idx in tscv.split(X):
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

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions = model.predict(X.iloc[test_idx])

        model_scores.append(
            accuracy_score(
                y.iloc[test_idx], predictions
            ) * 100
        )

        baseline_scores.append(
            y.iloc[test_idx].mean() * 100
        )

    results.append({
        "Stock": stock,
        "Model_Accuracy": np.mean(model_scores),
        "Baseline_Accuracy": np.mean(baseline_scores)
    })

results_df = pd.DataFrame(results)

results_df["Difference"] = (
    results_df["Model_Accuracy"]
    - results_df["Baseline_Accuracy"]
)

print("\n--- StockSense AI: All Stocks ---")
print(results_df.round(2).to_string(index=False))

results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

print("\nResults saved successfully!")
