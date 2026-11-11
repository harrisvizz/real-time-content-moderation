
import json
import numpy as np
import onnxruntime as ort

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


# Project paths
MODEL_PATH = "models/distilbert_toxicity_v3"
ONNX_PATH = "outputs/distilbert_toxicity_v3_onnx/model.onnx"
THRESHOLD_PATH = "outputs/v3_label_thresholds.json"

MAX_LENGTH = 128
REVIEW_THRESHOLD = 0.50

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# Initialize FastAPI
app = FastAPI(
    title="AI-Powered Content Moderation API",
    description="Real-time toxicity detection using DistilBERT and ONNX Runtime",
    version="1.0.0"
)


# Load model and tokenizer
print("Loading V3 ONNX moderation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

with open(THRESHOLD_PATH, "r") as file:
    thresholds = json.load(file)

print("Model and thresholds loaded successfully!")


# Request schema
class CommentRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=10000)


# Moderation function
def moderate_comment(comment):

    if not comment.strip():
        raise ValueError("Comment cannot be empty.")

    encoded = tokenizer(
        comment,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }

    logits = session.run(["logits"], inputs)[0][0]

    probabilities = 1.0 / (1.0 + np.exp(-logits))

    results = []

    for i, label in enumerate(LABEL_COLUMNS):

        score = float(probabilities[i])
        threshold = thresholds.get(label)

        if threshold is None:
            decision = (
                "HUMAN_REVIEW"
                if score >= REVIEW_THRESHOLD
                else "NO_AUTOMATIC_FLAG"
            )

        elif score >= threshold:
            decision = "FLAGGED"

        else:
            decision = "NO_AUTOMATIC_FLAG"

        results.append({
            "label": label,
            "score": round(score, 4),
            "threshold": threshold,
            "decision": decision
        })

    decisions = [item["decision"] for item in results]

    overall = (
        "REQUIRES MODERATION"
        if "FLAGGED" in decisions or "HUMAN_REVIEW" in decisions
        else "NO AUTOMATIC FLAG"
    )

    return {
        "comment": comment,
        "overall_decision": overall,
        "results": results
    }


# Health check endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "DistilBERT V3 ONNX",
        "runtime": "ONNX Runtime"
    }


# Moderation endpoint
@app.post("/moderate")
def moderate(request: CommentRequest):

    try:
        return moderate_comment(request.comment)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during moderation."
        )
