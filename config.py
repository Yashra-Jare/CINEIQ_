import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Limits (Useful for testing on your Mac without blowing up RAM)
# Set to None when you are ready to run the full 25M rows
SAMPLE_SIZE = 100000 

# Model Paths (Where we will save trained models)
MODEL_DIR = BASE_DIR / "models"
os.makedirs(MODEL_DIR, exist_ok=True)