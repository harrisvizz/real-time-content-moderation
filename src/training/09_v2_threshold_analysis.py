
import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from dataset import ToxicCommentDataset

MODEL_PATH = "models/distilbert_toxicity_v3"

MAX_LENGTH = 128
BATCH_SIZE = 8

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

THRESHOLDS = np.arange(0.05, 0.51, 0.01)

device = torch.device("cpu")

print("Loading validation dataset...")

val_df = pd.read_csv("data/processed/val_processed.csv")

val_df = val_df.sample(
    n=min(2000, len(val_df)),
    random_state=42
).reset_index(drop=True)

print("Validation samples:", len(val_df))

print("\nLoading V3 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

val_dataset = ToxicCommentDataset(val_df)


def collate_batch(batch):

    texts = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    encoded = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0
)

print("\nGenerating V2 predictions...")

all_probs = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        probabilities = torch.sigmoid(outputs.logits)

        all_probs.extend(probabilities.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

y_prob = np.array(all_probs)
y_true = np.array(all_labels)

best_thresholds = {}
results = []

print("\n======================================")
print("V2 PER-LABEL THRESHOLD ANALYSIS")
print("======================================")

for i, label in enumerate(LABEL_COLUMNS):

    actual = y_true[:, i]
    probabilities = y_prob[:, i]

    positive_count = int(actual.sum())

    if len(np.unique(actual)) < 2:
        print(f"\n{label}: Insufficient class variation")
        continue

    best_f1 = -1
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in THRESHOLDS:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            actual, predictions, zero_division=0
        )

        recall = recall_score(
            actual, predictions, zero_division=0
        )

        f1 = f1_score(
            actual, predictions, zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall

    best_thresholds[label] = best_threshold

    results.append({
        "label": label,
        "positive_samples": positive_count,
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1_score": best_f1
    })

    print(f"\nLabel: {label}")
    print("Positive samples:", positive_count)
    print("Best threshold:", round(best_threshold, 2))
    print("Precision:", round(best_precision, 4))
    print("Recall:", round(best_recall, 4))
    print("F1-score:", round(best_f1, 4))


os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/v3_label_thresholds.json", "w"
) as file:

    json.dump(best_thresholds, file, indent=4)

pd.DataFrame(results).to_csv(
    "outputs/v3_per_label_results.csv",
    index=False
)

print("\nV3 thresholds saved successfully!")

print("outputs/v3_label_thresholds.json")
print("outputs/v3_per_label_results.csv")

print("\nV3 THRESHOLD ANALYSIS COMPLETED!")
