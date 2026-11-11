
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import ToxicCommentDataset

MODEL_PATH = "models/distilbert_toxicity"
MAX_LENGTH = 128
BATCH_SIZE = 4

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Load a larger validation sample
print("Loading validation dataset...")

val_df = pd.read_csv("data/processed/val_processed.csv")

val_df = val_df.sample(
    n=min(2000, len(val_df)),
    random_state=42
).reset_index(drop=True)

print("Validation samples:", len(val_df))

# Load model and tokenizer
print("\nLoading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device("cpu")
model.to(device)
model.eval()

# Create dataset
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

# Generate probabilities
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

# Compare different thresholds
thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

print("\n========================================")
print("THRESHOLD COMPARISON")
print("========================================")

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

print("\n========================================")
print("THRESHOLD ANALYSIS COMPLETED!")
print("========================================")
