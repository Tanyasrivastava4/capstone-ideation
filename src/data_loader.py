#src/data_loader.py
#import pandas as pd
#import os

#def load_dataset(path="data/raw/patient_reports.csv"):
 #   if not os.path.exists(path):
  #      raise FileNotFoundError(f"Dataset not found at {path}")
  #  df = pd.read_csv(path)
    # Ensure minimal columns
 #   if "patient_id" not in df.columns or "report_text" not in df.columns:
  #      raise ValueError("CSV must contain 'patient_id' and 'report_text' columns")
    #Fill missing optional fields

  #  if "age" not in df.columns:
   #     df["age"] = None
   # if "gender" not in df.columns:
    #    df["gender"] = None
    
    #return df


# src/data_loader.py
"""
data_loader.py
--------------
Loads patient reports dataset from CSV with validation.
"""
import pandas as pd
import os

def load_dataset(path="data/raw/patient_reports.csv"):
    """
    Load patient reports CSV with basic validation.
    
    Args:
        path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Validated patient reports dataframe
    """
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at {path}")
    
    # Load CSV
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ["patient_id", "report_text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    # Fill optional columns with defaults
    if "age" not in df.columns:
        df["age"] = None
    if "gender" not in df.columns:
        df["gender"] = None
    if "urgency" not in df.columns:
        df["urgency"] = None
    if "department" not in df.columns:
        df["department"] = None
    
    print(f"✅ Loaded {len(df)} patient records from {path}")
    
    return df  # ✅ FIXED: Added return statement

# ✅ Test function
if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
