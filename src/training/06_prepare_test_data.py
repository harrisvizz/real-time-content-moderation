import os
import pandas as pd

VAL_PATH = "data/processed/val_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"

print("Loading validation dataset...")

val_df = pd.read_csv(VAL_PATH)

# Reproduce the earlier 200-sample evaluation set
used_200 = set(
    val_df.sample(
        n=min(200, len(val_df)),
        random_state=42
    ).index
)

# Reproduce the 2,000-sample threshold analysis set
used_2000 = set(
    val_df.sample(
        n=min(2000, len(val_df)),
        random_state=42
    ).index
)

# Exclude all previously used samples
used_indices = used_200.union(used_2000)

test_df = val_df.drop(index=used_indices).reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)

test_df.to_csv(TEST_PATH, index=False)

print("\n================================")
print("TEST DATASET PREPARATION")
print("================================")

print("Original validation samples:", len(val_df))
print("Previously used samples:", len(used_indices))
print("New test samples:", len(test_df))

print("\nTest dataset saved successfully!")
print(TEST_PATH)