
# AI-Powered Real-Time Content Moderation

An NLP-based content moderation system that detects toxic comments using a fine-tuned DistilBERT model, optimized with ONNX Runtime and deployed through FastAPI and Streamlit.

## Project Overview

This project identifies potentially harmful social media comments using multi-label text classification.

It detects six categories:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- ONNX Runtime
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy

## Model Details

- Base model: distilbert-base-uncased
- Training dataset: Jigsaw Toxic Comment Classification
- Training sample: 30,000 comments
- Training epochs: 1
- Maximum sequence length: 128
- Classification: Multi-label

## Model Evaluation

V3 test evaluation:

| Metric | Result |
|---|---:|
| Micro F1 | 0.6993 |
| Macro F1 | 0.5613 |
| Micro Precision | 0.5761 |
| Micro Recall | 0.8896 |
| Exact Match Accuracy | 0.8923 |

Note: The test set was used for multiple model comparisons, so these results are not an untouched final benchmark. Rare categories also have limited evaluation support.

## ONNX Optimization

| Runtime | Average inference |
|---|---:|
| PyTorch | 17.538 ms |
| ONNX Runtime | 7.240 ms |

Measured speedup: 2.42x.

These are preliminary CPU measurements for single-comment inference.

## Application Architecture

User Comment
    |
    v
Streamlit Frontend
    |
    v
FastAPI Backend
    |
    v
DistilBERT V3 ONNX Model
    |
    v
Toxicity Scores
    |
    v
Moderation Decisions

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start FastAPI

```bash
python -m uvicorn src.api.main:app --reload
```

### 3. Start Streamlit

Open another terminal:

```bash
python -m streamlit run app.py
```

### 4. Open the application

http://localhost:8501

API documentation:

http://127.0.0.1:8000/docs

## Project Limitations

- The model is a portfolio-scale prototype.
- Rare toxicity categories need more evaluation data.
- Predictions can contain false positives and false negatives.
- Flagged comments should be reviewed by humans.
- An unflagged comment is not guaranteed to be safe.

## Future Improvements

- Improve rare-category detection.
- Evaluate on an untouched external dataset.
- Add moderation history and analytics.
- Deploy the application to a cloud platform.
- Improve model performance and monitoring.

## Author

AI/ML and Data Science Project
