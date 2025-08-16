import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

class SmartFeatureEngineer:
    """
    Feature engineer that matches the features expected by the smart entry model.
    This generates the exact same features as train_smart_model.py
    """
    
    def __init__(self):
        pass
    
    def engineer_smart_features(self, df):
        """Engineer features that identify good entry points (matches train_smart_model.py)"""
        df = df.copy()
        
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
        
        return df
    
    def get_smart_features(self):
        """Return the exact features expected by the smart model"""
        return [
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
    
    def prepare_features_for_prediction(self, df):
        """
        Prepare features specifically for prediction.
        Returns DataFrame with only the features needed by the smart model.
        """
        # Engineer all features
        df_features = self.engineer_smart_features(df)
        
        # Get smart feature columns (exactly what the model expects)
        feature_cols = self.get_smart_features()
        
        # Ensure all features exist
        available_features = [f for f in feature_cols if f in df_features.columns]
        missing_features = [f for f in feature_cols if f not in df_features.columns]
        
        if missing_features:
            print(f"⚠️  Missing smart features: {missing_features}")
        
        # Return only feature columns, drop NaN rows
        result = df_features[available_features].dropna()
        
        return result