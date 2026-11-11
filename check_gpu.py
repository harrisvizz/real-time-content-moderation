
import torch
import platform

print("Python platform:", platform.platform())
print("PyTorch version:", torch.__version__)

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):",
          round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2))
else:
    print("No CUDA-compatible GPU detected.")
    print("Training will use CPU.")
