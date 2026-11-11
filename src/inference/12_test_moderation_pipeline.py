
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

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

print("Loading V3 ONNX moderation model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

with open(THRESHOLD_PATH, "r") as file:
    calibration = json.load(file)

thresholds = calibration

print("ONNX model and V3 thresholds loaded successfully!")


def moderate_comment(comment):

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
            if score >= REVIEW_THRESHOLD:
                decision = "HUMAN_REVIEW"
            else:
                decision = "NO_AUTOMATIC_FLAG"

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

    return results


# Test comments
test_comments = [
    "You are a wonderful person. Have a great day!",
    "You are stupid and absolutely worthless.",
    "I completely disagree with your opinion.",
    "I will hurt you if you come here again."
]

print("\n======================================")
print("REAL-TIME CONTENT MODERATION TEST")
print("======================================")

for comment in test_comments:

    print("\nComment:", comment)
    print("--------------------------------------")

    results = moderate_comment(comment)

    for result in results:
        print(
            f"{result['label']:15} | "
            f"Score: {result['score']:.4f} | "
            f"Decision: {result['decision']}"
        )

    decisions = [item["decision"] for item in results]

    if "FLAGGED" in decisions or "HUMAN_REVIEW" in decisions:
        print("Overall: REQUIRES MODERATION")
    else:
        print("Overall: NO AUTOMATIC FLAG")

print("\nONNX MODERATION PIPELINE TEST COMPLETED!")
