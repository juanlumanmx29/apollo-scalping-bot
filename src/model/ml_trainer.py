#!/usr/bin/env python3

"""
ML Model Trainer for Apollo Bot
Trains gradient boosting model to predict 0.4% price increases in 20 minutes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

class MLTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_info = {}
        
    def prepare_training_data(self, df, target_pct=0.4, periods=20):
        """Prepare data for training"""
        from .feature_engineer import FeatureEngineer
        
        # Create features
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df, target_pct, periods)
        
        # Get feature columns (exclude OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = df_features[feature_cols].fillna(0)
        y = df_features['target']
        
        # Remove rows where target is NaN
        valid_rows = ~y.isna()
        X = X[valid_rows]
        y = y[valid_rows]
        
        self.feature_names = feature_cols
        
        print(f"Training data prepared:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(X)}")
        print(f"  Positive samples: {y.sum()} ({y.mean():.1%})")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train gradient boosting model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining model...")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train gradient boosting model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            verbose=1
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        print(f"\nModel Performance:")
        print(f"  Train Accuracy: {train_score:.3f}")
        print(f"  Test Accuracy: {test_score:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return test_score
    
    def save_model(self, target_pct=0.4, periods=20, threshold=0.5):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Create model info
        self.model_info = {
            'model': self.model,
            'model_type': 'gradient_boost',
            'feature_names': self.feature_names,
            'target_percentage': target_pct,
            'prediction_periods': periods,
            'threshold': threshold,
            'trained_date': datetime.now().isoformat(),
            'n_features': len(self.feature_names)
        }
        
        # Save paths
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save with timestamp
        timestamped_path = f'models/apollo_model_gradient_boost_{timestamp}.joblib'
        joblib.dump(self.model_info, timestamped_path)
        
        # Save as latest
        latest_path = 'models/apollo_model_latest.joblib'
        joblib.dump(self.model_info, latest_path)
        
        print(f"\nModel saved:")
        print(f"  Timestamped: {timestamped_path}")
        print(f"  Latest: {latest_path}")
        
        return latest_path
    
    def load_data(self, data_file):
        """Load training data from CSV"""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"Loaded data: {len(df)} rows from {data_file}")
        
        return df

def main():
    """Train a new model"""
    trainer = MLTrainer()
    
    # Load data (use the most recent 1m data)
    data_file = 'data/ETHUSDT_1m_20250801.csv'
    df = trainer.load_data(data_file)
    
    # Prepare training data
    X, y = trainer.prepare_training_data(df, target_pct=0.4, periods=20)
    
    # Train model
    test_score = trainer.train_model(X, y)
    
    # Save model
    model_path = trainer.save_model(target_pct=0.4, periods=20, threshold=0.5)
    
    print(f"\n✅ Training complete!")
    print(f"   Test accuracy: {test_score:.3f}")
    print(f"   Model saved to: {model_path}")
    print(f"\nTo use the model, run: python3 apollo.py")

if __name__ == "__main__":
    main()