import pandas as pd

train = pd.read_csv("data/processed/train_processed.csv")
val = pd.read_csv("data/processed/val_processed.csv")

print("TRAINING DATA")
print(train.shape)
print(train.head())

print("\nVALIDATION DATA")
print(val.shape)
print(val.head())

print("\nTRAINING LABEL DISTRIBUTION")
print(train.iloc[:, 1:].sum())

print("\nVALIDATION LABEL DISTRIBUTION")
print(val.iloc[:, 1:].sum())