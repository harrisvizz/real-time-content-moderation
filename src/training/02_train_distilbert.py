
import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tqdm import tqdm

from dataset import ToxicCommentDataset

# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5

TRAIN_SAMPLES = 1000
VAL_SAMPLES = 200

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# CPU configuration
device = torch.device("cpu")
torch.set_num_threads(4)

print("Training device:", device)

# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv("data/processed/train_processed.csv")
val_df = pd.read_csv("data/processed/val_processed.csv")

# Select smaller subsets for initial CPU experiment
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

# -----------------------------------------
# 3. LOAD TOKENIZER
# -----------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------------------------
# 4. CREATE DATASETS
# -----------------------------------------

train_dataset = ToxicCommentDataset(train_df)
val_dataset = ToxicCommentDataset(val_df)


def collate_batch(batch):

    texts = [item[0] for item in batch]

    labels = torch.stack([
        item[1] for item in batch
    ])

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

# -----------------------------------------
# 5. LOAD DISTILBERT MODEL
# -----------------------------------------

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=6,
    problem_type="multi_label_classification"
)

model.to(device)

print("Model loaded successfully!")

# -----------------------------------------
# 6. LOSS FUNCTION AND OPTIMIZER
# -----------------------------------------

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------------------
# 7. TRAINING
# -----------------------------------------

print("\nStarting training...")

model.train()

total_loss = 0

progress_bar = tqdm(
    train_loader,
    desc="Training"
)

for batch in progress_bar:

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits

    loss = criterion(logits, labels)

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

    progress_bar.set_postfix(
        loss=f"{loss.item():.4f}"
    )

average_loss = total_loss / len(train_loader)

print("\nTraining completed!")
print("Average training loss:", round(average_loss, 4))

# -----------------------------------------
# 8. VALIDATION LOSS
# -----------------------------------------

print("\nCalculating validation loss...")

model.eval()

validation_loss = 0

with torch.no_grad():

    for batch in tqdm(val_loader, desc="Validation"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)

        validation_loss += loss.item()

average_val_loss = validation_loss / len(val_loader)

print("Validation loss:", round(average_val_loss, 4))

# -----------------------------------------
# 9. SAVE MODEL
# -----------------------------------------

os.makedirs("models/distilbert_toxicity", exist_ok=True)

model.save_pretrained("models/distilbert_toxicity")
tokenizer.save_pretrained("models/distilbert_toxicity")

print("\nModel saved successfully!")
print("Location: models/distilbert_toxicity")

print("\n====================================")
print("DISTILBERT TRAINING COMPLETED!")
print("====================================")
