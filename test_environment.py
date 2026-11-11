import torch
import transformers
import pandas
import sklearn
import onnxruntime
import fastapi
import streamlit

print("Environment setup successful!")

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("ONNX Runtime:", onnxruntime.__version__)