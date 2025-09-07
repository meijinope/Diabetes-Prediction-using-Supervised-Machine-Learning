import os, threading
import pandas as pd

DATA_FILE = "datasets/diabetes.csv"
REQUIRED_COLUMNS = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Diagnosis"
]

append_lock = threading.Lock()

def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    return df

def append_to_csv(row_dict):
    with append_lock:
        df_row = pd.DataFrame([row_dict])[REQUIRED_COLUMNS]
        file_exists = os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0
        df_row.to_csv(DATA_FILE, mode="a", header=not file_exists, index=False)
