#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MediPredict_Training')

class ModelTrainer:
    def __init__(self, data_dir='data/raw', model_dir='models'):
        """
        Initialize ModelTrainer with configurable data and model directories
        
        Args:
            data_dir (str): Directory containing raw data files
            model_dir (str): Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'preprocessor'), exist_ok=True)

    def load_dataset(self, filename):
        """
        Load dataset with error handling
        
        Args:
            filename (str): Name of the dataset file
        
        Returns:
            pd.DataFrame or None
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {filename}: {df.shape}")
            return df
        except FileNotFoundError:
            logger.warning(f"Dataset not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None

    def _create_preprocessing_pipeline(self, features):
        """
        Create a preprocessing pipeline for given features
        
        Args:
            features (list): List of feature column names
        
        Returns:
            ColumnTransformer
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, features)
            ],
            remainder='drop'
        )
        
        return preprocessor

    def _train_disease_model(self, X, y, disease_name, model_type='rf'):
        """
        Train a disease-specific model
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            disease_name (str): Name of the disease model
            model_type (str): Type of model to train
        
        Returns:
            dict: Trained model information
        """
        # Identify numeric features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ],
            remainder='drop'
        )
        
        # Select model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42
            )
        
        # Create full pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X[numeric_features], y, test_size=0.2, random_state=42
        )
        
        # Train model
        full_pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_score = full_pipeline.score(X_train, y_train)
        test_score = full_pipeline.score(X_test, y_test)
        
        logger.info(f"{disease_name.capitalize()} Model Training:")
        logger.info(f"Train Accuracy: {train_score:.4f}")
        logger.info(f"Test Accuracy: {test_score:.4f}")
        
        # Prepare model information
        model_info = {
            'model': full_pipeline,
            'preprocessor': preprocessor,  # Store preprocessor separately
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'features': numeric_features,
            'timestamp': datetime.now().isoformat()
        }
        
        return model_info

    def train_all_models(self):
        """
        Train models for all available disease datasets
        """
        datasets = {
            'diabetes': {
                'filename': 'diabetes.csv',
                'target_columns': ['target'],
                'rename_columns': {},
                'preprocessing': {
                    'handle_zeros': ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
                }
            },
            'heart_disease': {
                'filename': 'heart_disease.csv',
                'target_columns': ['target'],
                'rename_columns': {},
                'preprocessing': {}
            },
            'stroke': {
                'filename': 'stroke.csv',
                'target_columns': ['stroke'],
                'rename_columns': {
                    'avg_glucose_level': 'glucose',
                    'Residence_type': 'residence_type'
                },
                'preprocessing': {}
            },
            'framingham': {
                'filename': 'framingham_heart_study.csv',
                'target_columns': ['TenYearCHD'],
                'rename_columns': {
                    'sysBP': 'systolic_bp',
                    'diaBP': 'diastolic_bp',
                    'totChol': 'cholesterol',
                    'cigsPerDay': 'smoking_level'
                },
                'preprocessing': {}
            }
        }
        
        trained_models = {}
        
        for disease, config in datasets.items():
            logger.info(f"\n🔬 Training {disease.capitalize()} Model")
            
            # Load dataset
            df = self.load_dataset(config['filename'])
            
            if df is None:
                logger.warning(f"Skipping {disease} model training")
                continue
            
            # Rename columns if specified
            df.rename(columns=config['rename_columns'], inplace=True)
            
            # Preprocessing for specific datasets
            if 'handle_zeros' in config['preprocessing']:
                for col in config['preprocessing']['handle_zeros']:
                    if col in df.columns:
                        # Replace zeros with NaN for more robust imputation
                        df[col] = df[col].replace(0, np.nan)
            
            # Identify target column
            target_column = config['target_columns'][0]
            
            if target_column not in df.columns:
                logger.error(f"No target column found for {disease}. Available columns: {list(df.columns)}")
                continue
            
            logger.info(f"Using target column: {target_column}")
            
            # Prepare features and target
            try:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            except KeyError as e:
                logger.error(f"Error preparing data for {disease}: {e}")
                continue
            
            # Identify numeric features
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Add logging for feature selection
            logger.info(f"Numeric features for {disease}: {numeric_features}")
            
            # Ensure we have numeric features
            if not numeric_features:
                logger.error(f"No numeric features found for {disease}")
                continue
            
            # Subset X to only numeric features
            X = X[numeric_features]
            
            # Train model
            model_info = self._train_disease_model(X, y, disease)
            
            # Save model and preprocessor properly
            model_path = os.path.join(self.model_dir, f'{disease}_model.pkl')
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor', f'{disease}_preprocessor.pkl')
            
            # Option 1: Save just the pipeline (recommended)
            joblib.dump(model_info['model'], model_path)
            
            # Extract and save the preprocessor from the pipeline
            pipeline_preprocessor = model_info['model'].named_steps['preprocessor']
            joblib.dump(pipeline_preprocessor, preprocessor_path)
            
            # Option 2: Save model data as dictionary (if you need metadata)
            # joblib.dump({
            #     'pipeline': model_info['model'],
            #     'train_accuracy': model_info['train_accuracy'],
            #     'test_accuracy': model_info['test_accuracy'],
            #     'features': model_info['features'],
            #     'timestamp': model_info['timestamp']
            # }, model_path)
            
            logger.info(f"Saved {disease} model to {model_path}")
            logger.info(f"Saved {disease} preprocessor to {preprocessor_path}")
            
            # Verify saved files
            try:
                loaded_model = joblib.load(model_path)
                loaded_preprocessor = joblib.load(preprocessor_path)
                
                if hasattr(loaded_model, 'predict'):
                    logger.info(f"✓ {disease} model verified - has predict method")
                else:
                    logger.warning(f"⚠ {disease} model may be invalid - no predict method")
                
                if hasattr(loaded_preprocessor, 'transform'):
                    logger.info(f"✓ {disease} preprocessor verified - has transform method")
                else:
                    logger.warning(f"⚠ {disease} preprocessor may be invalid - no transform method")
                    
            except Exception as e:
                logger.error(f"Error verifying saved files for {disease}: {e}")
            
            trained_models[disease] = model_info
        
        return trained_models

def main():
    print("\n" + "="*60)
    print("🩺 MediPredict Model Training")
    print("="*60 + "\n")
    
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models()
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    for disease, model_info in trained_models.items():
        print(f"\n{disease.capitalize()} Model:")
        print(f"Train Accuracy: {model_info['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {model_info['test_accuracy']:.4f}")
        print(f"Features: {len(model_info['features'])} numeric features")
    
    print("\n🎉 Model Training Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()