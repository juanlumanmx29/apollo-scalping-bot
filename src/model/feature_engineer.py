import pandas as pd
import numpy as np
import ta

class FeatureEngineer:
    def __init__(self):
        self.features = []
    
    def add_technical_indicators(self, df):
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['price_momentum_10'] = df['close'].pct_change(10)
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        df['price_above_sma5'] = (df['close'] > df['sma_5']).astype(int)
        df['price_above_sma10'] = (df['close'] > df['sma_10']).astype(int)
        df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        
        # SMA distances
        df['dist_sma5'] = (df['close'] - df['sma_5']) / df['close'] * 100
        df['dist_sma10'] = (df['close'] - df['sma_10']) / df['close'] * 100
        df['dist_sma20'] = (df['close'] - df['sma_20']) / df['close'] * 100
        
        # Trend indicators
        df['sma5_trend'] = (df['sma_5'] > df['sma_5'].shift(1)).astype(int)
        df['sma10_trend'] = (df['sma_10'] > df['sma_10'].shift(1)).astype(int)
        
        return df
    
    def add_advanced_indicators(self, df):
        df = df.copy()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        return df
    
    def add_pattern_features(self, df):
        df = df.copy()
        
        # Candlestick patterns
        df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) & 
                       ((df['high'] - df['close']) < (df['close'] - df['open']))).astype(int)
        
        # Price action patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Gap detection
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        # Consecutive patterns
        df['consecutive_green'] = 0
        df['consecutive_red'] = 0
        
        green_candles = df['close'] > df['open']
        red_candles = df['close'] < df['open']
        
        for i in range(1, len(df)):
            if green_candles.iloc[i]:
                df.iloc[i, df.columns.get_loc('consecutive_green')] = df.iloc[i-1]['consecutive_green'] + 1 if green_candles.iloc[i-1] else 1
            if red_candles.iloc[i]:
                df.iloc[i, df.columns.get_loc('consecutive_red')] = df.iloc[i-1]['consecutive_red'] + 1 if red_candles.iloc[i-1] else 1
        
        return df
    
    def add_time_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Market session features (approximate for crypto)
        df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        
        return df
    
    def create_target(self, df, target_pct=0.4, periods=20):
        df = df.copy()
        future_returns = (df['close'].shift(-periods) / df['close'] - 1) * 100
        df['target'] = (future_returns >= target_pct).astype(int)
        return df
    
    def engineer_features(self, df, target_pct=0.4, periods=20):
        # print("Engineering features...")  # Disabled for batch testing
        
        df = self.add_technical_indicators(df)
        # print("✓ Basic technical indicators added")  # Silent for batch testing
        
        df = self.add_advanced_indicators(df)
        # print("✓ Advanced indicators added")  # Silent for batch testing
        
        df = self.add_pattern_features(df)
        # print("✓ Pattern features added")  # Silent for batch testing
        
        df = self.add_time_features(df)
        # print("✓ Time features added")  # Silent for batch testing
        
        df = self.create_target(df, target_pct, periods)
        # print("✓ Target variable created")  # Silent for batch testing
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        # print(f"✓ Cleaned data: {initial_rows} -> {final_rows} rows")  # Silent for batch testing
        
        # Store feature names (excluding OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        self.features = [col for col in df.columns if col not in exclude_cols]
        # print(f"✓ Created {len(self.features)} features")  # Silent for batch testing
        
        return df
    
    def prepare_features(self, df, target_pct=0.4, periods=20):
        """
        Alias for engineer_features for compatibility
        """
        return self.engineer_features(df, target_pct, periods)
    
    def get_feature_importance_data(self, df):
        return df[self.features + ['target']]