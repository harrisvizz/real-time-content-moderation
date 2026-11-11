
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ToxicCommentDataset

# Step 1: Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8

# Step 2: Load datasets
print("Loading processed datasets...")

train_df = pd.read_csv("data/processed/train_processed.csv")
val_df = pd.read_csv("data/processed/val_processed.csv")

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# Step 3: Load tokenizer
print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Step 4: Create PyTorch datasets
train_dataset = ToxicCommentDataset(train_df)
val_dataset = ToxicCommentDataset(val_df)

# Step 5: Define batch tokenization function
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

# Step 6: Create DataLoaders
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

# Step 7: Inspect one training batch
print("\nChecking first training batch...")

batch = next(iter(train_loader))

print("Input IDs shape:", batch["input_ids"].shape)
print("Attention Mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["labels"].shape)

print("\nFirst comment labels:")
print(batch["labels"][0])

print("\nDATA LOADER PREPARATION COMPLETED!")
