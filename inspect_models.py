#!/usr/bin/env python3
"""
Script to inspect what features the trained models expect
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSOR_DIR = os.path.join(MODEL_DIR, 'preprocessor')

# Model configurations
MODELS = {
    'diabetes': {
        'model_path': os.path.join(MODEL_DIR, 'diabetes_model.pkl'),
        'preprocessor_path': os.path.join(PREPROCESSOR_DIR, 'diabetes_preprocessor.pkl')
    },
    'heart_disease': {
        'model_path': os.path.join(MODEL_DIR, 'heart_disease_model.pkl'),
        'preprocessor_path': os.path.join(PREPROCESSOR_DIR, 'heart_disease_preprocessor.pkl')
    },
    'stroke': {
        'model_path': os.path.join(MODEL_DIR, 'stroke_model.pkl'),
        'preprocessor_path': os.path.join(PREPROCESSOR_DIR, 'stroke_preprocessor.pkl')
    },
    'framingham': {
        'model_path': os.path.join(MODEL_DIR, 'framingham_model.pkl'),
        'preprocessor_path': os.path.join(PREPROCESSOR_DIR, 'framingham_preprocessor.pkl')
    }
}

def get_feature_names_from_column_transformer(column_transformer):
    """Extract feature names from a ColumnTransformer"""
    feature_names = []
    
    # Get all transformers
    for name, transformer, columns in column_transformer.transformers_:
        if name == 'remainder':
            continue
            
        # If columns is a list of strings, use them directly
        if isinstance(columns, list):
            feature_names.extend(columns)
        # If columns is a slice or array of indices, we need the original feature names
        else:
            print(f"    Warning: Transformer '{name}' uses column indices, not names")
            feature_names.extend([f"feature_{i}" for i in range(len(columns))])
    
    return feature_names

def inspect_model(disease, paths):
    """Inspect a single model to understand its expected features"""
    print(f"\n{'='*60}")
    print(f"Inspecting {disease.upper()} Model")
    print('='*60)
    
    # Load model
    if os.path.exists(paths['model_path']):
        try:
            model_data = joblib.load(paths['model_path'])
            print(f"\n✓ Model loaded successfully")
            print(f"  Type: {type(model_data)}")
            
            # Extract pipeline if model is stored in a dictionary
            if isinstance(model_data, dict):
                print(f"  Model is stored as dictionary with keys: {list(model_data.keys())}")
                pipeline = model_data.get('pipeline') or model_data.get('model')
                if 'features' in model_data:
                    print(f"  Stored features: {model_data['features']}")
            else:
                pipeline = model_data
            
            # If it's a pipeline, inspect its steps
            if isinstance(pipeline, Pipeline):
                print(f"\n  Pipeline steps:")
                for step_name, step in pipeline.named_steps.items():
                    print(f"    - {step_name}: {type(step).__name__}")
                    
                    # If preprocessor is a ColumnTransformer, get its details
                    if isinstance(step, ColumnTransformer):
                        print(f"      Transformers:")
                        for name, transformer, columns in step.transformers_:
                            print(f"        - {name}: {type(transformer).__name__}")
                            if isinstance(columns, list):
                                print(f"          Columns: {columns}")
                            else:
                                print(f"          Column indices: {columns}")
                        
                        # Try to get feature names
                        try:
                            feature_names = get_feature_names_from_column_transformer(step)
                            print(f"\n      Expected features: {feature_names}")
                        except Exception as e:
                            print(f"      Could not extract feature names: {e}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    else:
        print(f"✗ Model file not found")
    
    # Load preprocessor
    print(f"\n{'-'*40}")
    if os.path.exists(paths['preprocessor_path']):
        try:
            preprocessor_data = joblib.load(paths['preprocessor_path'])
            print(f"✓ Preprocessor loaded successfully")
            print(f"  Type: {type(preprocessor_data)}")
            
            if isinstance(preprocessor_data, list):
                print(f"  ⚠️  Preprocessor is a list (likely feature names): {preprocessor_data}")
            elif isinstance(preprocessor_data, ColumnTransformer):
                print(f"  Preprocessor is a ColumnTransformer")
                try:
                    feature_names = get_feature_names_from_column_transformer(preprocessor_data)
                    print(f"  Expected features: {feature_names}")
                except Exception as e:
                    print(f"  Could not extract feature names: {e}")
            
        except Exception as e:
            print(f"✗ Error loading preprocessor: {e}")
    else:
        print(f"✗ Preprocessor file not found")

def create_test_predictions():
    """Try to make predictions with each model to see what features they expect"""
    print(f"\n\n{'='*60}")
    print("Testing Predictions to Identify Required Features")
    print('='*60)
    
    for disease, paths in MODELS.items():
        print(f"\n{disease.upper()}:")
        
        if not os.path.exists(paths['model_path']):
            print("  Model not found, skipping...")
            continue
            
        try:
            # Load model
            model_data = joblib.load(paths['model_path'])
            
            # Extract pipeline
            if isinstance(model_data, dict):
                pipeline = model_data.get('pipeline') or model_data.get('model')
            else:
                pipeline = model_data
            
            # Create a dummy DataFrame with many possible features
            dummy_features = {
                # Diabetes features
                'pregnancies': 0, 'glucose': 100, 'blood_pressure': 120,
                'skin_thickness': 20, 'insulin': 80, 'bmi': 25,
                'diabetes_pedigree': 0.5, 'age': 40,
                
                # Heart disease features
                'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 200,
                'fbs': 0, 'restecg': 0, 'thalach': 70, 'exang': 0,
                'oldpeak': 0, 'slope': 1, 'ca': 0, 'thal': 2,
                
                # Stroke features
                'gender': 'Male', 'hypertension': 0, 'heart_disease': 0,
                'ever_married': 'Yes', 'work_type': 'Private',
                'residence_type': 'Urban', 'avg_glucose_level': 100,
                'smoking_status': 'never smoked',
                
                # Framingham features
                'male': 1, 'education': 2, 'currentSmoker': 0,
                'smoking_level': 0, 'BPMeds': 0, 'prevalentStroke': 0,
                'prevalentHyp': 0, 'diabetes': 0, 'totChol': 200,
                'systolic_bp': 120, 'diastolic_bp': 80, 'BMI': 25,
                'heartRate': 70, 'cigsPerDay': 0
            }
            
            # Try prediction with all features
            df = pd.DataFrame([dummy_features])
            
            try:
                prediction = pipeline.predict(df)
                print(f"  ✓ Prediction successful with all features")
            except KeyError as e:
                print(f"  ✗ Missing required features: {e}")
                
                # Try to identify which features are needed
                if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                    preprocessor = pipeline.named_steps['preprocessor']
                    if hasattr(preprocessor, 'transformers_'):
                        for name, transformer, columns in preprocessor.transformers_:
                            if isinstance(columns, list):
                                print(f"    Required features from '{name}': {columns}")
            except Exception as e:
                print(f"  ✗ Prediction error: {e}")
                
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")

if __name__ == "__main__":
    print("MediPredict Model Inspector")
    print("="*60)
    
    # Inspect each model
    for disease, paths in MODELS.items():
        inspect_model(disease, paths)
    
    # Try test predictions
    create_test_predictions()
    
    print(f"\n\n{'='*60}")
    print("Inspection Complete")
    print("="*60)