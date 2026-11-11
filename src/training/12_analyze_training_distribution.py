
import pandas as pd

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

print("Loading complete training dataset...")

train_df = pd.read_csv("data/processed/train_processed.csv")

# Recreate the same 5,000-row sample used for V2 training
v2_sample = train_df.sample(
    n=5000,
    random_state=42
)

print("\n======================================")
print("TRAINING DATA DISTRIBUTION ANALYSIS")
print("======================================")

print("\nComplete training dataset:", len(train_df))
print("V2 training sample:", len(v2_sample))

print("\nPositive examples by category:\n")

for label in LABEL_COLUMNS:

    total_positive = int(train_df[label].sum())
    sample_positive = int(v2_sample[label].sum())

    print(f"{label}")
    print(f"  Complete dataset: {total_positive}")
    print(f"  V2 training sample: {sample_positive}")
    print(
        f"  Percentage retained: "
        f"{sample_positive / total_positive * 100:.2f}%"
        if total_positive > 0 else "  Percentage retained: N/A"
    )
    print()

print("DISTRIBUTION ANALYSIS COMPLETED!")
