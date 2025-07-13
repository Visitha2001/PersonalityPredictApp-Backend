import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from models.schemas import PersonalityInput, PersonalityPrediction

class PredictionController:
    def __init__(self):
        # Load trained model components
        model_components = joblib.load('final_ensemble_model.pkl')
        self.model_xgb = model_components['model_xgb']
        self.model_lgb = model_components['model_lgb']
        self.model_cat = model_components['model_cat']
        self.best_weights = model_components['best_weights']
        self.threshold = model_components['threshold_ensemble']
        self.le_target = model_components['le_target']
        self.scaler = model_components['scaler']
        self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        self.numerical_cols_no_target = model_components['numerical_cols_no_target']
        self.categorical_cols = model_components['categorical_cols']
        self.all_features_union = model_components['all_features_union']
        self.poly_feature_names = model_components['poly_feature_names']
        self.CONSTANT_FEATURE_NAME = 'constant_zero_feature'
        self.CONSTANT_FEATURE_VALUE = 0

    def preprocess_input(self, input_data: dict):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add constant feature
        input_df[self.CONSTANT_FEATURE_NAME] = self.CONSTANT_FEATURE_VALUE
        
        # Target encoding for categorical features
        for col in self.categorical_cols:
            if f'{col}_target_enc' in self.numerical_cols_no_target:
                input_df[f'{col}_target_enc'] = 0.5  # Default value if no mapping available
        
        # One-Hot Encoding
        input_df = pd.get_dummies(input_df, columns=self.categorical_cols, drop_first=False)
        
        # Ensure all expected columns are present
        for col in self.all_features_union:
            if col not in input_df.columns and col != 'id':
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[self.all_features_union]
        
        # Interaction terms
        if len(self.numerical_cols_no_target) >= 2:
            for i in range(min(2, len(self.numerical_cols_no_target))):
                for j in range(i + 1, min(2, len(self.numerical_cols_no_target))):
                    col1 = self.numerical_cols_no_target[i]
                    col2 = self.numerical_cols_no_target[j]
                    if col1 in input_df.columns and col2 in input_df.columns:
                        input_df[f'{col1}_{col2}_inter'] = input_df[col1] * input_df[col2]
        
        # Polynomial features
        if len(self.numerical_cols_no_target) >= 2:
            top_cols = self.numerical_cols_no_target[:2]
            if all(col in input_df.columns for col in top_cols):
                poly_features = self.poly.fit_transform(input_df[top_cols])
                for i, name in enumerate(self.poly_feature_names):
                    input_df[name] = poly_features[:, i]
        
        # Scale numerical features
        input_df[self.numerical_cols_no_target] = self.scaler.transform(input_df[self.numerical_cols_no_target])
        
        return input_df

    def predict(self, input_data: PersonalityInput) -> PersonalityPrediction:
        try:
            # Preprocess input
            features = self.preprocess_input(input_data.dict())
            
            # Make prediction
            xgb_proba = self.model_xgb.predict_proba(features)[:, 1]
            lgb_proba = self.model_lgb.predict_proba(features)[:, 1]
            cat_proba = self.model_cat.predict_proba(features)[:, 1]
            
            ensemble_proba = (self.best_weights['w_xgb'] * xgb_proba + 
                            self.best_weights['w_lgb'] * lgb_proba + 
                            self.best_weights['w_cat'] * cat_proba)
            
            prediction = (ensemble_proba > self.threshold).astype(int)
            personality = self.le_target.inverse_transform(prediction)[0]
            
            # Get confidence and indicators
            confidence = ensemble_proba[0] if personality == 'Extrovert' else 1 - ensemble_proba[0]
            indicators = self._get_indicators(input_data.dict())
            
            return PersonalityPrediction(
                prediction=personality,
                confidence=round(float(confidence), 2),
                indicators=indicators
            )
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def _get_indicators(self, input_data: dict) -> dict:
        indicators = {}
        if input_data['Time_spent_Alone'] > 6:
            indicators['high_alone_time'] = "Prefers significant alone time"
        if input_data['Social_event_attendance'] < 2:
            indicators['low_social_events'] = "Attends few social events"
        if input_data['Drained_after_socializing'] == 'Yes':
            indicators['drained'] = "Gets drained after socializing"
        return indicators