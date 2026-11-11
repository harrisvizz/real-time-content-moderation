import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/train.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
