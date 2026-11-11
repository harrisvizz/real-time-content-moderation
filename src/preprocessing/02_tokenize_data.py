
import pandas as pd
from transformers import AutoTokenizer

# Step 1: Load the processed training dataset
print("Loading processed dataset...")

train_df = pd.read_csv("data/processed/train_processed.csv")

print("Dataset shape:", train_df.shape)

# Step 2: Load the DistilBERT tokenizer
print("\nLoading DistilBERT tokenizer...")

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")

# Step 3: Select one sample comment
sample_text = train_df["comment_text"].iloc[0]

print("\nOriginal comment:")
print(sample_text)

# Step 4: Convert text into tokens
tokens = tokenizer.tokenize(sample_text)

print("\nTokens:")
print(tokens)

# Step 5: Convert tokens into numerical IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nToken IDs:")
print(token_ids)

# Step 6: Tokenize with special tokens,
# padding and truncation
encoded_input = tokenizer(
    sample_text,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

print("\nInput IDs:")
print(encoded_input["input_ids"])

print("\nAttention Mask:")
print(encoded_input["attention_mask"])

print("\nInput IDs shape:", encoded_input["input_ids"].shape)
print("Attention Mask shape:", encoded_input["attention_mask"].shape)

print("\nTOKENIZATION COMPLETED SUCCESSFULLY!")
