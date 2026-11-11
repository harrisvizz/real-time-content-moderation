
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

MODEL_PATH = "models/distilbert_toxicity_v2"

MAX_LENGTH = 128
BATCH_SIZE = 8

TARGET_PRECISION = 0.80
MIN_POSITIVE_SUPPORT = 50

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

THRESHOLDS = np.arange(0.05, 1.00, 0.01)

device = torch.device("cpu")

print("Loading validation dataset...")

val_df = pd.read_csv("data/processed/val_processed.csv")

val_df = val_df.sample(
    n=min(2000, len(val_df)),
    random_state=42
).reset_index(drop=True)

print("Validation samples:", len(val_df))

print("\nLoading V2 model...")

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

print("\nGenerating predictions...")

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

calibration = {
    "target_precision": TARGET_PRECISION,
    "minimum_positive_support": MIN_POSITIVE_SUPPORT,
    "thresholds": {},
    "manual_review_labels": {},
    "results": {}
}

print("\n======================================")
print("PRECISION-ORIENTED CALIBRATION")
print("======================================")

for i, label in enumerate(LABEL_COLUMNS):

    actual = y_true[:, i]
    probabilities = y_prob[:, i]

    positive_count = int(actual.sum())

    print(f"\nLabel: {label}")
    print("Positive samples:", positive_count)

    if positive_count < MIN_POSITIVE_SUPPORT:

        print("Insufficient positive examples.")
        print("Decision: Human review required.")

        calibration["thresholds"][label] = None
        calibration["manual_review_labels"][label] = (
            "Insufficient validation support"
        )
        continue

    best_result = None

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            actual,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0
        )

        if precision >= TARGET_PRECISION:

            if best_result is None or recall > best_result["recall"]:

                best_result = {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }

    if best_result is None:

        print("Target precision not achieved.")
        print("Decision: Human review required.")

        calibration["thresholds"][label] = None
        calibration["manual_review_labels"][label] = (
            "Target precision not achieved"
        )

    else:

        calibration["thresholds"][label] = best_result["threshold"]

        calibration["results"][label] = best_result

        print("Selected threshold:", round(best_result["threshold"], 2))
        print("Precision:", round(best_result["precision"], 4))
        print("Recall:", round(best_result["recall"], 4))
        print("F1-score:", round(best_result["f1_score"], 4))

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/v2_precision_thresholds.json", "w"
) as file:

    json.dump(calibration, file, indent=4)

print("\nCalibration saved successfully!")
print("outputs/v2_precision_thresholds.json")

print("\nPRECISION CALIBRATION COMPLETED!")

