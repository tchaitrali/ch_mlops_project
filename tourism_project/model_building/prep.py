import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get project root from environment variable
PROJECT_ROOT = os.environ.get("PROJECT_ROOT_DIR")
if not PROJECT_ROOT:
    raise RuntimeError("PROJECT_ROOT_DIR environment variable not set.")

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "tourism.csv")

df = pd.read_csv(RAW_PATH)

# Let's drop CustomerID as it's an unique identifier
df.drop(columns=["CustomerID"], inplace=True)

# Let's define X and target y
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits to CSV files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
