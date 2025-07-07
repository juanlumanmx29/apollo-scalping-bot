import requests
import pandas as pd
import numpy as np

def get_features_and_predict(model, symbol="ETHUSDT", threshold=0.73):
    """
    Descarga datos de Binance, calcula features y predice probabilidad usando el modelo dado.
    Devuelve: current_price, features_array, proba
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 100
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        # Features
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['BB_Middle'] = df['SMA_20']
        df['BB_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['RSI'] = df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean() / (
            df['close'].diff().abs().rolling(window=14).mean() + 1e-9) * 100
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
        df['Volatility'] = (df['high'] - df['low']) / df['low'] * 100
        df['Volatility_SMA'] = df['Volatility'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        latest = df.iloc[-1]
        current_price = latest['close']
        features = [
            latest['BB_Upper'], latest['high'], latest['EMA_20'], 
            latest['BB_Middle'], latest['SMA_20'], latest['close'], 
            latest['open'], latest['low'], latest['BB_Lower'],
            latest['RSI'], latest['Stoch_D'], latest['Stoch_K'], 
            latest['BB_Position'], latest['Volatility_SMA'], latest['ATR'], 
            latest['BB_Width'], latest['Volume_Ratio'], latest['Volatility'], 
            latest['volume']
        ]
        features = [0 if pd.isna(x) else x for x in features]
        features_array = np.array(features).reshape(1, -1)
        proba = float(model.predict_proba(features_array)[0, 1]) if model else 0.5
        return current_price, features_array, proba
    except Exception as e:
        return None, None, 0.0