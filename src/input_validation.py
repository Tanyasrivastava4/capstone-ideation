#src/input_validation.py

"""
input_validation.py
validates the input csv file before preprocessing
"""

import os
import pandas as pd

REQUIRED_COLUMNS = ["patient_id", "age", "gender","report_text"]

def validate_input_file(file_path: str) -> bool:
    """
    Check if the input CSV file exists and contains all required columns.
    Returns True if valid, false otherwise.
    """

    # File existance check
    if not os.path.exists(file_path):
        print(f"Error: File is not found at {file_path}")
        return False
    
    # loading csv
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading csv file : {e}")
        return False

    #checking columns
    missing_cols = [ col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        printf(f" Missing Required columns: {missing_cols}")
        return False
    
    # basic data sanity checks
    if df.empty:
        print("Error : The dataset is empty")
        return False
    
    if df["report_text"].isnull().any():
        print("Warning: some reports have missing test entries")
    
     # 5️⃣ Optional type checks
    if not pd.api.types.is_numeric_dtype(df["age"]):
        print("⚠️ Warning: 'age' column should be numeric.")

    print("✅ Input validation successful.")
    return True

if __name__ == "__main__":
    file_path = "data/raw/patient_reports.csv"
    valid = validate_input_file(file_path)
    if valid:
        print("Ready for next preprocessing step")
    else:
        print("Fix the above issues before proceeding.")
