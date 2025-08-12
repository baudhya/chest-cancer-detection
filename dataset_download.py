# https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code

import os
import zipfile
import requests
from tqdm import tqdm

# Dataset info - using Kaggle direct download URL
dataset_url = "https://www.kaggle.com/api/v1/datasets/download/mohamedhanyyy/chest-ctscan-images"
output_dir = "dataset"
zip_filename = "chest-ctscan-images.zip"

def download_file(url, filename):
    """
    Download a file from URL with progress bar
    """
    print(f"[INFO] Downloading dataset from Kaggle...")
    
    # Stream the download to show progress
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def extract_data_folder(zip_path, output_dir):
    """
    Extract contents from Data folder inside the ZIP file
    """
    print(f"[INFO] Extracting Data folder contents to ./{output_dir}")
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of all files in the ZIP
        file_list = zip_ref.namelist()
        
        # Find files that are in the Data folder
        data_files = [f for f in file_list if f.startswith('Data/')]
        
        if not data_files:
            print("[WARNING] No Data folder found in ZIP. Extracting all contents...")
            zip_ref.extractall(output_dir)
        else:
            # Extract files from Data folder, removing the Data/ prefix
            for file in data_files:
                # Skip the Data folder itself
                if file != 'Data/':
                    # Extract to output_dir with Data/ prefix removed
                    zip_ref.extract(file, output_dir)
                    
                    # Rename the extracted file to remove Data/ prefix
                    old_path = os.path.join(output_dir, file)
                    new_path = os.path.join(output_dir, file.replace('Data/', '', 1))
                    
                    if old_path != new_path:
                        # Create parent directory if it doesn't exist
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        # Move the file
                        os.rename(old_path, new_path)
            
            # Remove the empty Data folder
            data_folder = os.path.join(output_dir, 'Data')
            if os.path.exists(data_folder):
                try:
                    os.rmdir(data_folder)
                except OSError:
                    # Folder not empty, that's fine
                    pass

def main():
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download the dataset
        download_file(dataset_url, zip_filename)
        
        # Extract the Data folder from ZIP file
        extract_data_folder(zip_filename, output_dir)
        
        # Remove ZIP file after extraction
        os.remove(zip_filename)
        
        print(f"[INFO] Done. Dataset extracted to ./{output_dir}")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[INFO] Please download the dataset manually from:")
        print("https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images")
        print("[INFO] Extract it to the 'dataset' directory")
        
    except zipfile.BadZipFile:
        print(f"[ERROR] Invalid ZIP file: {zip_filename}")
        print("[INFO] Please check if the download was successful")
        
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
