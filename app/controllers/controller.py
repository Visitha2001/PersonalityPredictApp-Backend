from fastapi import HTTPException
import joblib
import numpy as np
import pandas as pd
from models.schemas import PersonalityInput

# Load models and metadata (will be initialized in main.py)
models = None
metadata = None

def initialize_models():
    global models, metadata
    try:
        models = {
            'xgb': joblib.load('model_xgb.joblib'),
            'lgb': joblib.load('model_lgb.joblib'),
            'cat': joblib.load('model_cat.joblib')
        }
        metadata = joblib.load('model_metadata.joblib')
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

def predict_personality(input_data: PersonalityInput):
    if not models or not metadata:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Add constant feature if used in training
        if 'constant_zero_feature' in metadata['FEATURES']:
            input_df['constant_zero_feature'] = 0
        
        # Ensure all required features are present
        for feature in metadata['FEATURES']:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        input_df = input_df[metadata['FEATURES']]
        
        # Get predictions from each model
        xgb_proba = models['xgb'].predict_proba(input_df)[:, 1]
        lgb_proba = models['lgb'].predict_proba(input_df)[:, 1]
        cat_proba = models['cat'].predict_proba(input_df)[:, 1]
        
        # Combine predictions with ensemble weights
        ensemble_proba = (metadata['best_weights']['w_xgb'] * xgb_proba +
                         metadata['best_weights']['w_lgb'] * lgb_proba +
                         metadata['best_weights']['w_cat'] * cat_proba)
        
        # Apply threshold
        prediction = (ensemble_proba > metadata['threshold_ensemble']).astype(int)
        
        # Convert to label
        return metadata['le_target'].inverse_transform(prediction)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")