import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

class ScalpingFeatureEngineer:
    """
    Feature engineer specifically for scalping (1-5 minute movements).
    Generates the exact features needed by the XGBoost scalping model.
    """
    
    def __init__(self):
        self.features = []
    
    def engineer_scalping_features(self, df):
        """Engineer features specifically for scalping (1-5 minute movements)"""
        df = df.copy()
        
        # Basic price features
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_3m'] = df['close'].pct_change(3)
        df['price_change_5m'] = df['close'].pct_change(5)
        
        # NEW: Mean reversion features (critical for avoiding FOMO)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['distance_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['sma_50'] = df['close'].rolling(50).mean()
        df['distance_from_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # NEW: Position in range (to detect if we're at highs)
        df['high_20'] = df['close'].rolling(20).max()
        df['low_20'] = df['close'].rolling(20).min()
        df['position_in_range'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)
        
        # NEW: Pullback indicators
        df['pullback_from_high_5'] = (df['high'].rolling(5).max() - df['close']) / df['close']
        df['pullback_from_high_10'] = (df['high'].rolling(10).max() - df['close']) / df['close']
        
        # Micro-structure features (very important for scalping)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Volume features (critical for detecting real moves)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # Buy/Sell pressure
        df['buy_pressure'] = df['volume'] * df['close_position']
        df['sell_pressure'] = df['volume'] * (1 - df['close_position'])
        df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-10)
        
        # Momentum features (short-term)
        df['momentum_1m'] = df['close'] - df['close'].shift(1)
        df['momentum_3m'] = df['close'] - df['close'].shift(3)
        df['momentum_5m'] = df['close'] - df['close'].shift(5)
        df['momentum_acceleration'] = df['momentum_1m'] - df['momentum_1m'].shift(1)
        
        # Technical indicators (fast versions for scalping)
        # RSI - very short period
        df['rsi_6'] = RSIIndicator(df['close'], window=6).rsi()
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
        
        # MACD - faster settings
        macd = MACD(df['close'], window_slow=12, window_fast=6, window_sign=4)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands - tighter
        bb = BollingerBands(df['close'], window=10, window_dev=1.5)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # ATR for volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=7)
        df['atr'] = atr.average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Recent high/low
        df['dist_from_high_5m'] = (df['high'].rolling(5).max() - df['close']) / df['close']
        df['dist_from_low_5m'] = (df['close'] - df['low'].rolling(5).min()) / df['close']
        
        # Price patterns (micro)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['bullish_pattern'] = df['higher_high'] & df['higher_low']
        
        # Order flow imbalance (approximation)
        df['delta'] = df['close'] - df['open']
        df['cumulative_delta'] = df['delta'].rolling(10).sum()
        
        # Time features (some periods are better for scalping)
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        # Market regime
        df['volatility_regime'] = df['atr_ratio'].rolling(60).mean()
        df['trend_strength'] = abs(df['price_change_5m'].rolling(20).mean())
        
        return df
    
    def get_scalping_features(self):
        """Return the list of features used for scalping"""
        return [
            # Price action
            'price_change_1m', 'price_change_3m', 'price_change_5m',
            
            # NEW: Mean reversion features
            'distance_from_sma20', 'distance_from_sma50',
            'position_in_range', 'pullback_from_high_5', 'pullback_from_high_10',
            'high_low_ratio', 'close_position', 'upper_shadow', 'lower_shadow',
            
            # Volume
            'volume_ratio', 'volume_trend', 'volume_spike',
            'buy_pressure', 'sell_pressure', 'pressure_ratio',
            
            # Momentum
            'momentum_1m', 'momentum_3m', 'momentum_5m', 'momentum_acceleration',
            
            # Technical
            'rsi_6', 'rsi_14', 'macd', 'macd_signal', 'macd_diff',
            'bb_position', 'atr_ratio',
            
            # Market structure
            'dist_from_high_5m', 'dist_from_low_5m',
            'higher_high', 'higher_low', 'bullish_pattern',
            'delta', 'cumulative_delta',
            
            # Time
            'hour', 'is_london_session', 'is_ny_session', 'is_asian_session',
            
            # Regime
            'volatility_regime', 'trend_strength'
        ]
    
    def create_target(self, df, target_pct=0.003, periods=5):
        """Create target: will price move up by target_pct within next periods?"""
        df = df.copy()
        
        # Calculate future prices
        future_prices = []
        for i in range(1, periods + 1):
            future_prices.append(df['close'].shift(-i))
        
        # Get max price in next 'periods' minutes
        future_max = pd.concat(future_prices, axis=1).max(axis=1)
        
        # Target: will price reach target_pct gain?
        df['target'] = ((future_max - df['close']) / df['close'] >= target_pct).astype(int)
        
        return df
    
    def engineer_features(self, df, target_pct=0.3, periods=5):
        """
        Main method to engineer features for scalping model.
        Kept for compatibility with existing interface.
        """
        # Convert percentage to decimal if needed
        target_pct_decimal = target_pct / 100 if target_pct > 1 else target_pct
        
        # Engineer scalping features
        df = self.engineer_scalping_features(df)
        
        # Create target if needed
        if 'target' not in df.columns:
            df = self.create_target(df, target_pct_decimal, periods)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Store feature names
        self.features = self.get_scalping_features()
        
        return df
    
    def prepare_features_for_prediction(self, df):
        """
        Prepare features specifically for prediction (without target).
        Returns DataFrame with only the features needed by the scalping model.
        """
        # Engineer all features
        df_features = self.engineer_scalping_features(df)
        
        # Get scalping feature columns
        feature_cols = self.get_scalping_features()
        
        # Ensure all features exist
        available_features = [f for f in feature_cols if f in df_features.columns]
        
        # Return only feature columns, drop NaN rows
        result = df_features[available_features].dropna()
        
        return result