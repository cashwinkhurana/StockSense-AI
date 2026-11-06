
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load Apple's historical data
df = pd.read_csv(
    "data/features/AAPL_features.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date")

# Features used for prediction

features = [
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

target = "Target_5D_Return"

# Exclude rows without known future returns
df = df.dropna(subset=features + [target])

# Chronological 80/20 split (no random shuffling)
split = int(len(df) * 0.8)

# Remove 5 training rows near the test boundary
# to prevent overlapping future target information
train = df.iloc[:split - 5]
test = df.iloc[split:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Create XGBoost regression model
model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Generate predictions on unseen data
predictions = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

direction_accuracy = (
    (predictions > 0) == (y_test.to_numpy() > 0)
).mean() * 100

print("StockSense AI - AAPL Model Results")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Mean Absolute Error: {mae:.2f}%")
print(f"Root Mean Squared Error: {rmse:.2f}%")
print(f"Direction Accuracy: {direction_accuracy:.2f}%")


# Baseline 1: Always predict a 0% return
baseline_predictions = np.zeros(len(y_test))

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

# Baseline 2: Always predict an upward movement
up_accuracy = (y_test.to_numpy() > 0).mean() * 100

print("\n--- Baseline Comparison ---")
print(f"XGBoost MAE: {mae:.2f}%")
print(f"Zero-Return Baseline MAE: {baseline_mae:.2f}%")
print(f"XGBoost Direction Accuracy: {direction_accuracy:.2f}%")
print(f"Always-Up Direction Accuracy: {up_accuracy:.2f}%")

# Save trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/AAPL_xgboost.pkl")

print("Model saved successfully!")
