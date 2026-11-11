
import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataset import ToxicCommentDataset

# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

MODEL_PATH = "models/distilbert_toxicity"

MAX_LENGTH = 128
BATCH_SIZE = 4
THRESHOLD = 0.5

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

device = torch.device("cpu")

# -----------------------------------------
# 2. LOAD VALIDATION DATA
# -----------------------------------------

print("Loading validation dataset...")

val_df = pd.read_csv("data/processed/val_processed.csv")

# Use the same 200 validation samples used during training
val_df = val_df.sample(
    n=min(200, len(val_df)),
    random_state=42
).reset_index(drop=True)

print("Validation samples:", len(val_df))

# -----------------------------------------
# 3. LOAD TOKENIZER AND MODEL
# -----------------------------------------

print("\nLoading trained DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully!")

# -----------------------------------------
# 4. CREATE DATASET AND DATALOADER
# -----------------------------------------

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


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0
)

# -----------------------------------------
# 5. GENERATE PREDICTIONS
# -----------------------------------------

print("\nGenerating predictions...")

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.sigmoid(outputs.logits)

        predictions = (
            probabilities >= THRESHOLD
        ).int()

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.numpy())


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

# -----------------------------------------
# 6. CALCULATE EVALUATION METRICS
# -----------------------------------------

print("\n====================================")
print("MODEL EVALUATION RESULTS")
print("====================================")

print("\nExact Match Accuracy:")
print(accuracy_score(y_true, y_pred))

print("\nMicro Precision:")
print(precision_score(
    y_true, y_pred, average="micro", zero_division=0
))

print("\nMicro Recall:")
print(recall_score(
    y_true, y_pred, average="micro", zero_division=0
))

print("\nMicro F1 Score:")
print(f1_score(
    y_true, y_pred, average="micro", zero_division=0
))

print("\nMacro F1 Score:")
print(f1_score(
    y_true, y_pred, average="macro", zero_division=0
))

# -----------------------------------------
# 7. PER-LABEL CLASSIFICATION REPORT
# -----------------------------------------

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    target_names=LABEL_COLUMNS,
    zero_division=0
)

print(report)

# -----------------------------------------
# 8. ROC-AUC
# -----------------------------------------

print("\nROC-AUC Scores:")

auc_scores = {}

for i, label in enumerate(LABEL_COLUMNS):

    if len(np.unique(y_true[:, i])) == 2:

        auc = roc_auc_score(
            y_true[:, i],
            y_prob[:, i]
        )

        auc_scores[label] = auc

        print(f"{label}: {auc:.4f}")

    else:
        print(f"{label}: Cannot calculate - only one class present")

# -----------------------------------------
# 9. SAVE RESULTS
# -----------------------------------------

os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame({
    "Actual": list(y_true),
    "Predicted": list(y_pred)
})

results.to_csv(
    "outputs/model_predictions.csv",
    index=False
)

print("\nPredictions saved to outputs/model_predictions.csv")

print("\n====================================")
print("MODEL EVALUATION COMPLETED!")
print("====================================")
