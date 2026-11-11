
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "thesofakillers/jigsaw-toxic-comment-classification-challenge"

os.makedirs("data/raw", exist_ok=True)

print("Starting Jigsaw dataset download...")

try:
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename="train.csv",
        repo_type="dataset"
    )

    destination = "data/raw/train.csv"

    shutil.copy2(file_path, destination)

    print("Dataset downloaded successfully!")
    print("Location:", destination)

except Exception as e:
    print("Download failed!")
    print("Error:", repr(e))
