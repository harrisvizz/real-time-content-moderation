
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

from dataset import ToxicCommentDataset

MODEL_PATH = "models/distilbert_toxicity_v3"
TEST_PATH = "data/processed/test_processed.csv"
THRESHOLD_PATH = "outputs/v3_label_thresholds.json"

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

device = torch.device("cpu")

print("Loading test dataset...")
test_df = pd.read_csv(TEST_PATH)
print("Test samples:", len(test_df))

print("\nLoading V2 thresholds...")
with open(THRESHOLD_PATH, "r") as file:
    thresholds = json.load(file)

threshold_values = np.array([
    thresholds[label] for label in LABEL_COLUMNS
])

print("Thresholds:", thresholds)

print("\nLoading V2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

test_dataset = ToxicCommentDataset(test_df)


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


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0
)

print("\nGenerating V2 test predictions...")

all_probs = []
all_labels = []

with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        probabilities = torch.sigmoid(outputs.logits)

        all_probs.extend(probabilities.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {(batch_index + 1) * BATCH_SIZE} "
                f"/ {len(test_dataset)} comments"
            )

y_prob = np.array(all_probs)
y_true = np.array(all_labels)

# Apply V2 individual thresholds
y_pred = (y_prob >= threshold_values).astype(int)

print("\n======================================")
print("V2 FINAL TEST RESULTS")
print("======================================")

print("Test samples:", len(y_true))

print(
    "Exact Match Accuracy:",
    round(accuracy_score(y_true, y_pred), 4)
)

print(
    "Micro Precision:",
    round(precision_score(
        y_true, y_pred, average="micro", zero_division=0
    ), 4)
)

print(
    "Micro Recall:",
    round(recall_score(
        y_true, y_pred, average="micro", zero_division=0
    ), 4)
)

print(
    "Micro F1 Score:",
    round(f1_score(
        y_true, y_pred, average="micro", zero_division=0
    ), 4)
)

print(
    "Macro F1 Score:",
    round(f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ), 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=LABEL_COLUMNS,
        zero_division=0
    )
)

print("\nROC-AUC:")

for i, label in enumerate(LABEL_COLUMNS):

    try:
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        print(f"{label}: {auc:.4f}")

    except ValueError:
        print(f"{label}: Cannot calculate")

# Save predictions
os.makedirs("outputs", exist_ok=True)

prediction_df = pd.DataFrame()

for i, label in enumerate(LABEL_COLUMNS):

    prediction_df[f"{label}_actual"] = y_true[:, i]
    prediction_df[f"{label}_probability"] = y_prob[:, i]
    prediction_df[f"{label}_predicted"] = y_pred[:, i]

prediction_df.to_csv(
    "outputs/v3_test_predictions.csv",
    index=False
)

print("\nV3 predictions saved to:")
print("outputs/v3_test_predictions.csv")

print("\nV3 TEST EVALUATION COMPLETED!")
