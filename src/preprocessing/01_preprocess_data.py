
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/raw/train.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# STEP 2: DEFINE TARGET LABELS
# ==========================================

label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ==========================================
# STEP 3: HANDLE MISSING VALUES
# ==========================================

print("\nChecking missing values...")

print(df.isnull().sum())

# Remove rows without actual comments
df = df.dropna(subset=["comment_text"]).copy()

# Remove rows with missing target labels
df = df.dropna(subset=label_columns).copy()

print("\nShape after removing missing values:")
print(df.shape)


# ==========================================
# STEP 4: REMOVE DUPLICATE RECORDS
# ==========================================

print("\nDuplicate IDs before removal:")
print(df["id"].duplicated().sum())

df = df.drop_duplicates(subset=["id"]).copy()

print("Shape after duplicate removal:")
print(df.shape)


# ==========================================
# STEP 5: TEXT NORMALIZATION
# ==========================================

def clean_text(text):

    # Convert input to string
    text = str(text)

    # Remove unnecessary leading/trailing spaces
    text = text.strip()

    # Replace multiple spaces and newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text


df["comment_text"] = df["comment_text"].apply(clean_text)


# ==========================================
# STEP 6: REMOVE EMPTY COMMENTS
# ==========================================

df = df[df["comment_text"] != ""].copy()

print("\nShape after text preprocessing:")
print(df.shape)


# ==========================================
# STEP 7: TEXT LENGTH ANALYSIS
# ==========================================

df["text_length"] = df["comment_text"].apply(
    lambda text: len(text.split())
)

print("\nText length statistics:")
print(df["text_length"].describe())


# ==========================================
# STEP 8: SEPARATE INPUT AND LABELS
# ==========================================

X = df["comment_text"]

y = df[label_columns]

print("\nInput shape:", X.shape)
print("Label shape:", y.shape)


# ==========================================
# STEP 9: TRAIN-VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# STEP 10: SAVE PROCESSED DATASETS
# ==========================================

os.makedirs("data/processed", exist_ok=True)

train_df = pd.concat(
    [
        X_train.rename("comment_text"),
        y_train
    ],
    axis=1
)

val_df = pd.concat(
    [
        X_val.rename("comment_text"),
        y_val
    ],
    axis=1
)

train_df.to_csv(
    "data/processed/train_processed.csv",
    index=False
)

val_df.to_csv(
    "data/processed/val_processed.csv",
    index=False
)


# ==========================================
# STEP 11: FINAL SUMMARY
# ==========================================

print("\n====================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("====================================")

print("\nFinal training dataset:", train_df.shape)
print("Final validation dataset:", val_df.shape)

print("\nSaved files:")
print("data/processed/train_processed.csv")
print("data/processed/val_processed.csv")
