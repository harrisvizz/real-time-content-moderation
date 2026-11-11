
import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from torch.optim import AdamW
from sklearn.metrics import f1_score

from dataset import ToxicCommentDataset

# Configuration
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_PATH = "models/distilbert_toxicity_v3"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

TRAIN_SAMPLES = 30000
VAL_SAMPLES = 2000

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

device = torch.device("cpu")

print("Loading datasets...")

train_df = pd.read_csv("data/processed/train_processed.csv")
val_df = pd.read_csv("data/processed/val_processed.csv")

train_df = train_df.sample(
    n=min(TRAIN_SAMPLES, len(train_df)),
    random_state=42
).reset_index(drop=True)

val_df = val_df.sample(
    n=min(VAL_SAMPLES, len(val_df)),
    random_state=42
).reset_index(drop=True)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# Calculate positive class weights
label_counts = train_df[LABEL_COLUMNS].sum().values
negative_counts = len(train_df) - label_counts

pos_weights = negative_counts / np.maximum(label_counts, 1)

# Prevent extreme weights
pos_weights = np.clip(pos_weights, 1.0, 20.0)

pos_weights = torch.tensor(
    pos_weights,
    dtype=torch.float32
).to(device)

print("\nPositive class weights:")
for label, weight in zip(LABEL_COLUMNS, pos_weights):
    print(label, ":", round(weight.item(), 2))

print("\nLoading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    problem_type="multi_label_classification"
)

model.to(device)

# Dataset
train_dataset = ToxicCommentDataset(train_df)
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


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0
)

# Optimizer and weighted loss
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)

print("\nStarting V2 training...")

for epoch in range(EPOCHS):

    model.train()
    total_train_loss = 0

    for batch_index, batch in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

        loss = criterion(
            outputs.logits,
            batch["labels"].to(device)
        )

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in val_loader:

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )

            labels = batch["labels"].to(device)

            loss = criterion(outputs.logits, labels)

            total_val_loss += loss.item()

            probabilities = torch.sigmoid(outputs.logits)

            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    y_pred = (y_prob >= 0.5).astype(int)

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    print("\nEpoch:", epoch + 1)
    print("Training loss:", round(avg_train_loss, 4))
    print("Validation loss:", round(avg_val_loss, 4))
    print("Validation micro F1:", round(micro_f1, 4))

# Save model
os.makedirs(OUTPUT_PATH, exist_ok=True)

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\nV2 model saved successfully!")
print("Location:", OUTPUT_PATH)
