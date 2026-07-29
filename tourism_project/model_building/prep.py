import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Construct path to tourism.csv robustly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Go up from model_building to tourism_project
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
