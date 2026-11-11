
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/raw/train.csv")

# Define toxicity categories
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ======================================
# GRAPH 1: TOXICITY CATEGORY DISTRIBUTION
# ======================================

label_counts = df[labels].sum()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=label_counts.index,
    y=label_counts.values
)

plt.title("Distribution of Toxicity Categories")
plt.xlabel("Toxicity Categories")
plt.ylabel("Number of Comments")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("outputs/toxicity_distribution.png")

plt.show()


# ======================================
# GRAPH 2: TOXIC VS NON-TOXIC COMMENTS
# ======================================

df["total_labels"] = df[labels].sum(axis=1)

toxic = (df["total_labels"] > 0).sum()
safe = (df["total_labels"] == 0).sum()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Non-toxic", "Toxic"],
    [safe, toxic]
)

plt.title("Toxic vs Non-Toxic Comments")
plt.xlabel("Comment Type")
plt.ylabel("Number of Comments")

plt.tight_layout()

plt.savefig("outputs/toxic_vs_safe.png")

plt.show()

print("Both visualizations generated successfully!")
