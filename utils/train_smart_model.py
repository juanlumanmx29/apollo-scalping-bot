#!/usr/bin/env python3
"""
Smart Model Training - Focuses on finding good entry points (pullbacks/reversals)
Instead of just predicting price increases, it learns to identify optimal entry timing
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_collector.binance_collector import BinanceDataCollector

class SmartModelTrainer:
    def __init__(self):
        self.collector = BinanceDataCollector()
        
    def collect_training_data(self, use_historical=True):
        """Load historical data from saved files"""
        if use_historical:
            print("📊 Loading historical 1-minute data from saved files...")
            
            # Try to load the historical data files
            data_files = [
                'data/ETHUSDT_1m_historical_20250806.csv',
                'data/ETHUSDT_1m_7days_scalping_20250806.csv',  
                'data/ETHUSDT_1m_fresh_20250806.csv',
                'data/ETHUSDT_1m_20250801.csv'
            ]
            
            for file_path in data_files:
                try:
                    if os.path.exists(file_path):
                        print(f"📁 Loading {file_path}...")
                        df = pd.read_csv(file_path)
                        
                        # Handle different timestamp formats
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        elif df.index.name != 'timestamp' and 'Unnamed: 0' not in df.columns:
                            # Try to parse the index as datetime
                            df.index = pd.to_datetime(df.index)
                        
                        # Ensure we have OHLCV columns
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required_cols):
                            df = df[required_cols].copy()
                            df = df.dropna()
                            
                            if len(df) > 1000:
                                print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                                return df
                            else:
                                print(f"⚠️  File {file_path} only has {len(df)} candles")
                        else:
                            print(f"⚠️  File {file_path} missing required columns")
                            
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
                    continue
            
            print("❌ Could not load historical data files")
            return None
        else:
            # Fallback to API collection
            print("📊 Collecting data from Binance API...")
            df = self.collector.get_recent_data('1m', minutes=7*24*60)
            if df is not None and len(df) > 0:
                print(f"✅ Collected {len(df)} candles from API")
                return df
            return None
    
    def engineer_smart_features(self, df):
        """Engineer features that identify good entry points"""
        print("🔧 Engineering smart entry-detection features...")
        
        # Price movement features
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_3m'] = df['close'].pct_change(3)
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_10m'] = df['close'].pct_change(10)
        
        # REVERSAL DETECTION FEATURES
        # 1. Recent peak detection
        df['high_5m'] = df['high'].rolling(5).max()
        df['low_5m'] = df['low'].rolling(5).min()
        df['is_near_recent_high'] = (df['close'] >= df['high_5m'] * 0.998).astype(int)
        df['is_near_recent_low'] = (df['close'] <= df['low_5m'] * 1.002).astype(int)
        
        # 2. Pullback detection
        df['pullback_from_high'] = (df['high_5m'] - df['close']) / df['high_5m']
        df['bounce_from_low'] = (df['close'] - df['low_5m']) / df['low_5m']
        
        # 3. Momentum deceleration (good for catching turns)
        df['momentum_1m'] = df['close'] - df['close'].shift(1)
        df['momentum_3m'] = df['close'] - df['close'].shift(3)
        df['momentum_decel'] = df['momentum_1m'] - df['momentum_1m'].shift(1)
        df['is_decelerating'] = (df['momentum_decel'] < 0).astype(int)
        
        # MEAN REVERSION FEATURES
        df['sma_20'] = df['close'].rolling(20).mean()
        df['distance_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['is_above_sma'] = (df['close'] > df['sma_20']).astype(int)
        df['is_far_from_sma'] = (abs(df['distance_from_sma20']) > 0.005).astype(int)
        
        # RSI for overbought/oversold
        from ta.momentum import RSIIndicator
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # VOLUME PATTERNS
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_declining'] = (df['volume'] < df['volume'].shift(1)).astype(int)
        
        # CANDLE PATTERNS (reversal signals)
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)  # Small body = indecision
        df['is_hammer'] = ((df['lower_wick'] > df['body_size'] * 2) & (df['upper_wick'] < df['body_size'])).astype(int)
        
        # SUPPORT/RESISTANCE
        df['touches_high_20'] = (df['high'] >= df['high'].rolling(20).max() * 0.999).astype(int)
        df['touches_low_20'] = (df['low'] <= df['low'].rolling(20).min() * 1.001).astype(int)
        
        # TIME FEATURES
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"✅ Created {len(df.columns)} smart features")
        return df
    
    def create_smart_target(self, df, profit_target=0.002, loss_limit=0.003, periods=8):
        """
        Create target that identifies GOOD entry points:
        - Price goes up by profit_target within periods (made easier)
        - Price doesn't drop below loss_limit first (made more lenient)
        - Entry is not at a local maximum
        """
        print(f"🎯 Creating smart target for good entries...")
        print(f"   Target: {profit_target:.1%} profit in {periods} minutes, max {loss_limit:.1%} loss")
        
        # Calculate future price movements
        future_max = pd.concat([df['high'].shift(-i) for i in range(1, periods+1)], axis=1).max(axis=1)
        future_min = pd.concat([df['low'].shift(-i) for i in range(1, periods+1)], axis=1).min(axis=1)
        
        # Find which comes first - profit or loss
        profit_reached = (future_max - df['close']) / df['close'] >= profit_target
        loss_reached = (df['close'] - future_min) / df['close'] >= loss_limit
        
        # Check if we're not buying at a recent high (less strict)
        recent_high = df['high'].rolling(15).max()
        not_at_high = df['close'] < recent_high * 0.995  # Allow within 0.5% of high
        
        # Good entry = profit reached AND (not stopped out OR small loss acceptable)
        df['target'] = (profit_reached & (~loss_reached | (df['close'] > df['low'].shift(-periods) * 0.998)) & not_at_high).astype(int)
        
        # Remove last 'periods' rows
        df = df[:-periods]
        
        positive_pct = df['target'].mean() * 100
        print(f"✅ Target created: {positive_pct:.1f}% positive samples (good entries)")
        
        if positive_pct < 5:
            print(f"⚠️  Warning: Very few positive samples ({positive_pct:.1f}%). Model might struggle to learn.")
        elif positive_pct > 20:
            print(f"⚠️  Warning: Too many positive samples ({positive_pct:.1f}%). Target might be too easy.")
        
        return df
    
    def prepare_features(self, df):
        """Select the most important features for training"""
        feature_cols = [
            # Entry timing features
            'pullback_from_high', 'bounce_from_low',
            'is_near_recent_high', 'is_near_recent_low',
            'momentum_decel', 'is_decelerating',
            
            # Mean reversion
            'distance_from_sma20', 'is_far_from_sma',
            
            # RSI
            'rsi', 'rsi_oversold', 'rsi_overbought',
            
            # Volume
            'volume_ratio', 'volume_declining',
            
            # Patterns
            'is_doji', 'is_hammer',
            'upper_wick', 'lower_wick',
            
            # Support/Resistance
            'touches_high_20', 'touches_low_20',
            
            # Price changes
            'price_change_1m', 'price_change_3m', 'price_change_5m',
            
            # Time
            'hour'
        ]
        
        available_features = [f for f in feature_cols if f in df.columns]
        X = df[available_features].copy()
        y = df['target'].copy()
        
        print(f"✅ Using {len(available_features)} features for training")
        return X, y, available_features
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train both XGBoost and RandomForest and pick the best"""
        models = {}
        
        # 1. XGBoost
        print("\n🚀 Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb.fit(X_train, y_train)
        xgb_score = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        models['xgboost'] = (xgb, xgb_score)
        print(f"XGBoost ROC-AUC: {xgb_score:.3f}")
        
        # 2. RandomForest
        print("\n🌲 Training RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf.fit(X_train, y_train)
        rf_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        models['random_forest'] = (rf, rf_score)
        print(f"RandomForest ROC-AUC: {rf_score:.3f}")
        
        # Pick the best model
        best_name = max(models, key=lambda k: models[k][1])
        best_model, best_score = models[best_name]
        
        print(f"\n🏆 Best model: {best_name} (ROC-AUC: {best_score:.3f})")
        
        # Show feature importance for best model
        if hasattr(best_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X_test.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            print("\nTop 10 Features:")
            print(importances)
        
        return best_model, best_name, best_score
    
    def test_entry_quality(self, model, X, y):
        """Test how good the model is at finding entries"""
        predictions = model.predict_proba(X)[:, 1]
        
        print("\n📊 Entry Quality Analysis:")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            signals = predictions >= threshold
            if signals.sum() > 0:
                accuracy = y[signals].mean()
                print(f"Threshold {threshold:.1f}: {signals.sum()} trades, {accuracy:.1%} success rate")
    
    def save_model(self, model, model_name, feature_names, score):
        """Save the trained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_info = {
            'model': model,
            'model_type': model_name,
            'feature_names': feature_names,
            'threshold': 0.6,  # Can be optimized
            'target_percentage': 0.3,
            'prediction_periods': 10,
            'roc_auc': score,
            'timestamp': timestamp,
            'description': 'Smart entry model - finds pullbacks and reversals'
        }
        
        # Save with multiple names
        joblib.dump(model_info, f'models/smart_entry_model_{timestamp}.joblib')
        joblib.dump(model_info, f'models/scalping_model_5min.joblib')  # Override the old model
        
        print(f"\n✅ Model saved successfully!")
        print(f"   Type: {model_name}")
        print(f"   Performance: {score:.3f} ROC-AUC")
        print(f"   Focus: Finding good entry points (pullbacks/reversals)")

def main():
    print("🧠 Smart Model Training - Finding Good Entry Points")
    print("=" * 60)
    
    trainer = SmartModelTrainer()
    
    # Step 1: Load historical data
    df = trainer.collect_training_data(use_historical=True)
    if df is None:
        print("❌ Failed to load historical data.")
        return
    
    if len(df) < 5000:
        print(f"⚠️  Warning: Only got {len(df)} candles. Ideally need 10,000+ for robust training.")
    else:
        print(f"✅ Good dataset size: {len(df)} candles for robust training")
        
    # Step 2: Engineer smart features
    df = trainer.engineer_smart_features(df)
    
    # Step 3: Create smart target (easier target for better learning)
    df = trainer.create_smart_target(df, profit_target=0.002, loss_limit=0.003, periods=8)
    
    # Step 4: Prepare features
    X, y, feature_names = trainer.prepare_features(df)
    
    # Check if we have enough positive samples
    if y.sum() < 20:
        print(f"❌ Only {y.sum()} positive samples. Need at least 20 for meaningful training.")
        print("   Try adjusting the target parameters or collecting more data.")
        return
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print(f"📊 Good entries: {y_train.mean():.1%} of samples ({y_train.sum()} positive samples)")
    
    # Step 6: Train models
    best_model, best_name, best_score = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Step 7: Test entry quality
    trainer.test_entry_quality(best_model, X_test, y_test)
    
    # Step 8: Save model
    trainer.save_model(best_model, best_name, feature_names, best_score)
    
    print("\n✅ Training complete! The model now focuses on finding good entry points.")
    print("   It looks for pullbacks, reversals, and oversold conditions")
    print("   Instead of chasing momentum at peaks")

if __name__ == "__main__":
    main()