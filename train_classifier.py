
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib

# Load stock data
df = pd.read_csv(
    "data/features/AAPL_features.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date")


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

# Remove rows with unknown targets
df = df.dropna(
    subset=features + ["Target_5D_Return"]
)

# Classification target: 1 = UP, 0 = DOWN/FLAT
df["Target_Direction"] = (
    df["Target_5D_Return"] > 0
).astype(int)

# Chronological train/test split
split = int(len(df) * 0.8)

# Purge 5 observations near the boundary
train = df.iloc[:split - 5]
test = df.iloc[split:]

X_train = train[features]
y_train = train["Target_Direction"]

X_test = test[features]
y_test = test["Target_Direction"]

# Train classification model

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


# Compare training and testing performance
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(
    y_train, train_predictions
) * 100

test_accuracy = accuracy_score(
    y_test, test_predictions
) * 100

print("\n--- Overfitting Analysis ---")
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")
print(f"Accuracy Gap: {train_accuracy - test_accuracy:.2f}%")


# Predict stock direction
predictions = model.predict(X_test)


# Analyse predicted probabilities
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- Prediction Probability Analysis ---")
print(f"Minimum UP probability: {probabilities.min():.4f}")
print(f"Maximum UP probability: {probabilities.max():.4f}")
print(f"Average UP probability: {probabilities.mean():.4f}")

print("\nPrediction Distribution:")
print(f"Predicted UP: {(predictions == 1).sum()}")
print(f"Predicted DOWN/FLAT: {(predictions == 0).sum()}")

print("\nFeature Importance:")
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance.head(10))


# Evaluate model
accuracy = accuracy_score(y_test, predictions) * 100
baseline = y_test.mean() * 100

print("\nStockSense AI - Classification Results")
print(f"Model Accuracy: {accuracy:.2f}%")
print(f"Always-Up Baseline: {baseline:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["DOWN/FLAT", "UP"],
    zero_division=0
))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(
    model,
    "models/AAPL_classifier.pkl"
)

print("Classification model saved successfully!")
