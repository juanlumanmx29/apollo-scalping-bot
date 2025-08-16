import joblib
import pandas as pd
import numpy as np
from src.model.feature_engineer import FeatureEngineer
from src.model.scalping_feature_engineer import ScalpingFeatureEngineer
from src.model.smart_feature_engineer import SmartFeatureEngineer
import os

class PricePredictor:
    def __init__(self, model_path="models/apollo_model_latest.joblib"):
        self.model_path = model_path
        self.model_info = None
        self.feature_engineer = FeatureEngineer()
        self.scalping_feature_engineer = ScalpingFeatureEngineer()
        self.smart_feature_engineer = SmartFeatureEngineer()
        self.is_scalping_model = False
        self.is_smart_model = False
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model_info = joblib.load(self.model_path)
            
            # Detect model type based on features and path
            self.is_smart_model = self._detect_smart_model()
            self.is_scalping_model = self._detect_scalping_model() and not self.is_smart_model
            
            if self.is_smart_model:
                model_type_display = f"{self.model_info['model_type']} (smart entry)"
            elif self.is_scalping_model:
                model_type_display = f"{self.model_info['model_type']} (scalping)"
            else:
                model_type_display = self.model_info['model_type']
            print(f"✓ Model loaded: {model_type_display}")
            print(f"  Threshold: {self.model_info['threshold']}")
            print(f"  Target: {self.model_info['target_percentage']}% in {self.model_info['prediction_periods']} periods")
            
            if self.is_smart_model:
                print(f"  Features: {len(self.model_info['feature_names'])} smart entry features")
            elif self.is_scalping_model:
                print(f"  Features: {len(self.model_info['feature_names'])} scalping features")
                
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def _detect_scalping_model(self):
        """Detect if this is a scalping model based on features and model type"""
        if not self.model_info or 'feature_names' not in self.model_info:
            return False
        
        feature_names = self.model_info['feature_names']
        scalping_features = self.scalping_feature_engineer.get_scalping_features()
        
        # Check if most features match scalping features
        scalping_feature_count = sum(1 for f in feature_names if f in scalping_features)
        scalping_ratio = scalping_feature_count / len(feature_names) if feature_names else 0
        
        # Consider it a scalping model if:
        # 1. XGBoost model AND has scalping-specific features, OR
        # 2. Model path contains "scalping", OR
        # 3. >80% of features are scalping features
        is_xgb_with_scalping_features = (
            self.model_info.get('model_type') == 'xgboost' and 
            any(f in feature_names for f in ['price_change_1m', 'buy_pressure', 'volume_trend'])
        )
        
        has_scalping_in_path = 'scalping' in self.model_path.lower()
        high_scalping_ratio = scalping_ratio > 0.8
        
        return is_xgb_with_scalping_features or has_scalping_in_path or high_scalping_ratio
    
    def _detect_smart_model(self):
        """Detect if this is the new smart entry model"""
        if not self.model_info or 'feature_names' not in self.model_info:
            return False
        
        feature_names = self.model_info['feature_names']
        smart_features = self.smart_feature_engineer.get_smart_features()
        
        # Check if most features match smart features
        smart_feature_count = sum(1 for f in feature_names if f in smart_features)
        smart_ratio = smart_feature_count / len(feature_names) if feature_names else 0
        
        # Check for smart model indicators
        has_smart_features = any(f in feature_names for f in ['pullback_from_high', 'bounce_from_low', 'is_decelerating'])
        has_smart_description = self.model_info.get('description', '').lower().find('smart entry') != -1
        
        # Consider it a smart model if:
        # 1. Has key smart features like pullback_from_high, OR
        # 2. Description mentions "smart entry", OR  
        # 3. >80% of features match smart features
        return has_smart_features or has_smart_description or smart_ratio > 0.8
    
    def prepare_features(self, df):
        """Prepare features using the appropriate feature engineer"""
        if self.is_smart_model:
            # Use smart feature engineer for the new smart entry model
            df_features = self.smart_feature_engineer.prepare_features_for_prediction(df)
        elif self.is_scalping_model:
            # Use scalping feature engineer
            df_features = self.scalping_feature_engineer.prepare_features_for_prediction(df)
        else:
            # Use regular feature engineer
            df_features = self.feature_engineer.engineer_features(
                df, 
                self.model_info['target_percentage'], 
                self.model_info['prediction_periods']
            )
        
        # Get only the features used in training
        feature_cols = self.model_info['feature_names']
        
        # Ensure all required features are available
        available_features = [f for f in feature_cols if f in df_features.columns]
        missing_features = [f for f in feature_cols if f not in df_features.columns]
        
        if missing_features:
            print(f"⚠️  Warning: Missing features: {missing_features}")
        
        if not available_features:
            print("❌ No required features found in data")
            return pd.DataFrame()
        
        # Get only the latest row with available features
        X = df_features[available_features].tail(1)
        
        return X
    
    def predict(self, df):
        # Smart models need enough candles for rolling calculations (SMA20, RSI14, etc.)
        if self.is_smart_model:
            min_candles = 60  # Need at least 60 for SMA20 and other rolling features
        elif self.is_scalping_model:
            min_candles = 60  # For scalping models 
        else:
            min_candles = 50  # For regular models
        
        if len(df) < min_candles:
            return None, 0.0
        
        try:
            X = self.prepare_features(df)
            
            if X.empty or X.isnull().any().any():
                print("⚠️  Warning: Features contain NaN values")
                return None, 0.0
            
            model = self.model_info['model']
            
            # Handle different model types
            if self.model_info['model_type'] == 'logistic' and 'scaler' in self.model_info:
                # Logistic regression with scaling
                X_scaled = self.model_info['scaler'].transform(X)
                prediction_proba = model.predict_proba(X_scaled)[0]
            elif self.model_info['model_type'] == 'neural_network':
                # Neural network (would need special handling, but not implemented yet)
                print("⚠️  Neural network models not supported yet")
                return None, 0.0
            elif self.model_info['model_type'] in ['xgboost', 'random_forest', 'gradient_boost']:
                # Tree-based models (including XGBoost)
                prediction_proba = model.predict_proba(X)[0]
            else:
                # Default: assume sklearn-compatible model
                prediction_proba = model.predict_proba(X)[0]
            
            confidence = prediction_proba[1]  # Probability of positive class
            
            should_buy = confidence >= self.model_info['threshold']
            
            return should_buy, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def get_prediction_details(self):
        if not self.model_info:
            return {}
        
        return {
            'model_type': self.model_info['model_type'],
            'threshold': self.model_info['threshold'],
            'target_percentage': self.model_info['target_percentage'],
            'prediction_periods': self.model_info['prediction_periods'],
            'timestamp': self.model_info.get('timestamp', 'Unknown')
        }