import pandas as pd
import os

# Get project root from environment variable
PROJECT_ROOT = os.environ.get("PROJECT_ROOT_DIR")
if not PROJECT_ROOT:
    raise RuntimeError("PROJECT_ROOT_DIR environment variable not set.")

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "tourism.csv")

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation","MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
