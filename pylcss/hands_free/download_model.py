# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Vosk Model Downloader.

Downloads and extracts the Vosk speech recognition model.
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

# Model info
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def download_vosk_model(progress_callback=None):
    """
    Download and extract the Vosk model.
    
    Args:
        progress_callback: Optional callback(downloaded, total) for progress updates
        
    Returns:
        Path to the extracted model directory, or None on failure
    """
    model_path = MODELS_DIR / MODEL_NAME
    
    # Check if already downloaded
    if model_path.exists() and (model_path / "am" / "final.mdl").exists():
        print(f"Model already exists at {model_path}")
        return model_path
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = MODELS_DIR / f"{MODEL_NAME}.zip"
    
    try:
        print(f"Downloading Vosk model from {MODEL_URL}...")
        print("This may take a few minutes (~40 MB)...")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            downloaded = count * block_size
            if progress_callback:
                progress_callback(downloaded, total_size)
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                print(f"\rDownloading: {percent}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="", flush=True)
        
        urllib.request.urlretrieve(MODEL_URL, zip_path, reporthook)
        print("\nDownload complete!")
        
        # Extract
        print(f"Extracting to {MODELS_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        
        # Clean up zip
        os.remove(zip_path)
        print(f"Model extracted to {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nPlease download manually from:")
        print(f"  {MODEL_URL}")
        print(f"And extract to:")
        print(f"  {MODELS_DIR}")
        
        # Clean up partial download
        if zip_path.exists():
            os.remove(zip_path)
            
        return None


def check_model() -> bool:
    """Check if the Vosk model is available."""
    model_path = MODELS_DIR / MODEL_NAME
    return model_path.exists() and (model_path / "am" / "final.mdl").exists()


def get_model_path() -> Path:
    """Get the path to the Vosk model."""
    return MODELS_DIR / MODEL_NAME


if __name__ == "__main__":
    print("Vosk Model Downloader for PyLCSS Hands-Free Control")
    print("=" * 50)
    
    if check_model():
        print(f"✓ Model already installed at:\n  {get_model_path()}")
    else:
        result = download_vosk_model()
        if result:
            print(f"\n✓ Model successfully installed!")
        else:
            print(f"\n✗ Model installation failed.")
            sys.exit(1)
