import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Helper to normalize column names

def normalize_columns(df: pd.DataFrame):
    """Standardize column names to lowercase, strips, and underscores."""
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace('[\s\-]+', '_', regex=True)
    )
    return df

# Download & normalize each dataset from its URL, then save raw CSV for further processing

def download_uci_heart_disease():
    """Download and save normalized UCI Heart Disease dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, header=None, names=cols, na_values='?')
    df = normalize_columns(df)
    df['target'] = (df['target'] > 0).astype(int)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/heart_disease.csv", index=False)
    return df


def download_pima_diabetes():
    """Download and save normalized Pima Indians Diabetes dataset."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age", "target"]
    df = pd.read_csv(url, header=None, names=cols)
    df = normalize_columns(df)
    # Replace invalid zeros
    for c in ['glucose','blood_pressure','skin_thickness','insulin','bmi']:
        df[c] = df[c].replace(0, np.nan)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/diabetes.csv", index=False)
    return df


def download_stroke_prediction():
    """Download and save normalized Stroke Prediction dataset."""
    url = ("https://storage.googleapis.com/kagglesdsdata/datasets/1120859/1882037/"
           "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(url)
    df = normalize_columns(df)
    # Clean
    df = df.replace({'n/a': np.nan, 'na': np.nan, 'unknown': np.nan})
    # Rename residence column if misnamed
    if 'residence_type' not in df.columns and 'residence_type' in df.columns:
        df = df.rename(columns={'residence_type': 'residence_type'})
    # Drop id
    df = df.drop(columns=['id'], errors='ignore')
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/stroke.csv", index=False)
    return df


def download_framingham_cardiovascular():
    """Download and save normalized Framingham dataset."""
    url = ("https://storage.googleapis.com/kagglesdsdata/datasets/6344673/10256448/"
           "framingham_heart_study.csv")
    df = pd.read_csv(url)
    df = normalize_columns(df)
    # Rename TenYearCHD to target
    if 'tenyearchd' in df.columns:
        df = df.rename(columns={'tenyearchd': 'target'})
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/cardiovascular_risk.csv", index=False)
    return df


def download_synthetic_patient_data():
    """Download and save synthetic patient data with normalized columns."""
    url = ("https://storage.googleapis.com/kagglesdsdata/datasets/3756889/6499491/all_prevalences.csv")
    df = pd.read_csv(url)
    df = normalize_columns(df)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_patient_data.csv", index=False)
    return df


def prepare_datasets():
    """Retrieve all raw CSVs and generate processed splits."""
    # Step 1: Raw downloads
    heart = download_uci_heart_disease()
    pima = download_pima_diabetes()
    stroke = download_stroke_prediction()
    cardio = download_framingham_cardiovascular()
    synth = download_synthetic_patient_data()

    # Step 2: Processing & splits
    os.makedirs("data/processed", exist_ok=True)
    all_sets = {
        'heart_disease': heart,
        'diabetes': pima,
        'stroke': stroke,
        'cardiovascular': cardio,
        'synthetic': synth
    }
    for name, df in all_sets.items():
        tv, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(tv, test_size=0.25, random_state=42)
        train.to_csv(f"data/processed/{name}_train.csv", index=False)
        val.to_csv(f"data/processed/{name}_val.csv", index=False)
        test.to_csv(f"data/processed/{name}_test.csv", index=False)
        print(f"{name}: train {train.shape}, val {val.shape}, test {test.shape}")
    print("Datasets ready.")


if __name__ == "__main__":
    prepare_datasets()
