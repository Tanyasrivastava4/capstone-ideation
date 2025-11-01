#src/data_loader.py
import pandas as pd
import os


def load_dataset(path="data/raw/patient_reports.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Ensure minimal columns
    if "patient_id" not in df.columns or "report_text" not in df.columns:
        raise ValueError("CSV must contain 'patient_id' and 'report_text' columns")
    #Fill missing optional fields

    if "age" not in df.columns:
        df["age"] = None
    if "gender" not in df.columns:
        df["gender"] = None
    
    return df