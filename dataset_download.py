# https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code


import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Dataset info
dataset_slug = "mohamedhanyyy/chest-ctscan-images"
output_dir = "dataset"
zip_filename = dataset_slug.split("/")[-1] + ".zip"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize Kaggle API
print("[INFO] Initializing Kaggle API...")
api = KaggleApi()
api.authenticate()

# Download the dataset
print(f"[INFO] Downloading dataset: {dataset_slug}")
api.dataset_download_files(dataset=dataset_slug, path=".", quiet=False)

# Extract the ZIP file
print(f"[INFO] Extracting {zip_filename} to ./{output_dir}")
with zipfile.ZipFile(zip_filename, "r") as zip_ref:
    zip_ref.extractall(output_dir)

# Remove ZIP file after extraction
os.remove(zip_filename)

print(f"[INFO] Done. Dataset extracted to ./{output_dir}")
