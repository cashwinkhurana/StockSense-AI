
import pandas as pd

df = pd.read_csv("data/features/AAPL_features.csv")

# Remove rows with unknown future returns
df = df.dropna(subset=["Target_5D_Return"])

# Create direction labels
df["Direction"] = (
    df["Target_5D_Return"] > 0
).astype(int)

# Match the existing training/test split
split = int(len(df) * 0.8)

train = df.iloc[:split - 5]
test = df.iloc[split:]

print("\n--- Training Data ---")
print("Total samples:", len(train))
print("UP:", (train["Direction"] == 1).sum())
print("DOWN/FLAT:", (train["Direction"] == 0).sum())
print("UP percentage:",
      round(train["Direction"].mean() * 100, 2))

print("\n--- Testing Data ---")
print("Total samples:", len(test))
print("UP:", (test["Direction"] == 1).sum())
print("DOWN/FLAT:", (test["Direction"] == 0).sum())
print("UP percentage:",
      round(test["Direction"].mean() * 100, 2))
