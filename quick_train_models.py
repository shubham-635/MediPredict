import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import logging

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def create_directories():
    """Create required directories"""
    for folder in ['models', 'models/preprocessor', 'data/processed']:
        os.makedirs(folder, exist_ok=True)

def load_data():
    """Load datasets from raw data"""
    datasets = {}
    raw_data_paths = {
        "diabetes": "data/raw/diabetes.csv",
        "heart_disease": "data/raw/heart_disease.csv",
        "framingham": "data/raw/framingham_heart_study.csv",
        "stroke": "data/raw/stroke.csv"
    }

    for name, path in raw_data_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                datasets[name] = df
                logger.info(f"Loaded {name} data with shape: {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        else:
            logger.warning(f"Data file for {name} not found: {path}")
    return datasets

def build_preprocessor(numeric_features):
    """Create a preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

def train_model(df, label_column, model_name):
    """Train a model for a specific dataset"""
    if label_column not in df.columns:
        logger.warning(f"Label column {label_column} not found in dataset. Skipping {model_name}")
        return None

    df = df.copy()

    # Remove rows with missing labels
    df = df.dropna(subset=[label_column])

    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Detect numeric columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    preprocessor = build_preprocessor(numeric_features)

    # Preprocess features
    X_transformed = preprocessor.fit_transform(X)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_transformed, y)

    # Save model
    model_data = {
        'model': model,
        'feature_names': numeric_features,
        'label': label_column
    }

    model_path = f"models/{model_name}_model.pkl"
    joblib.dump(model_data, model_path)
    logger.info(f"Saved model: {model_path}")

    # Save preprocessor
    preprocessor_path = f"models/preprocessor/{model_name}_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor: {preprocessor_path}")

    # Report accuracy on training data
    accuracy = model.score(X_transformed, y)
    logger.info(f"{model_name.capitalize()} model training accuracy: {accuracy:.3f}")

    return model

def main():
    print("\n" + "="*60)
    print("MediPredict - Quick Train All Models")
    print("="*60 + "\n")

    create_directories()
    datasets = load_data()

    # Dataset-specific label columns
    targets = {
        'diabetes': 'target',
        'heart_disease': 'target',
        'framingham': 'TenYearCHD',
        'stroke': 'stroke'
    }

    trained_models = {}
    for name, df in datasets.items():
        label = targets.get(name)
        if label:
            model = train_model(df, label, name)
            trained_models[name] = model

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    for name in trained_models:
        print(f"✓ Model saved: models/{name}_model.pkl")
        print(f"✓ Preprocessor saved: models/preprocessor/{name}_preprocessor.pkl")
    print("\nYou can now start the MediPredict app using these models.\n" + "="*60)

if __name__ == "__main__":
    main()
