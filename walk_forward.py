
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Load historical stock data
df = pd.read_csv(
    "data/features/NVDA_features.csv",
    parse_dates=["Date"]
)

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

df = df.sort_values("Date")
df = df.dropna(
    subset=features + ["Target_5D_Return"]
)

X = df[features]
y = (df["Target_5D_Return"] > 0).astype(int)

# Five chronological validation folds
# Gap prevents overlapping 5-day targets
tscv = TimeSeriesSplit(
    n_splits=5,
    gap=5
)

model_scores = []
baseline_scores = []

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(X), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

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

    predictions = model.predict(X_test)

    model_accuracy = accuracy_score(
        y_test, predictions
    ) * 100

    # Always-UP baseline
    baseline_accuracy = y_test.mean() * 100

    model_scores.append(model_accuracy)
    baseline_scores.append(baseline_accuracy)

    print(
        f"Fold {fold}: "
        f"Model={model_accuracy:.2f}% | "
        f"Baseline={baseline_accuracy:.2f}%"
    )

print("\n--- Walk-Forward Results ---")
print(
    f"Average Model Accuracy: "
    f"{np.mean(model_scores):.2f}%"
)
print(
    f"Average Baseline Accuracy: "
    f"{np.mean(baseline_scores):.2f}%"
)
