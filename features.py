import requests
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_current_price_simple(symbol="ETHUSDT"):
    """Simple price fetch for fallback when full features fail"""
    try:
        urls = [
            "https://api.binance.com/api/v3/ticker/price",
            "https://api1.binance.com/api/v3/ticker/price", 
            "https://api2.binance.com/api/v3/ticker/price"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            try:
                response = requests.get(url, params={"symbol": symbol}, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                return float(data["price"])
            except Exception as e:
                logger.debug(f"Price fetch failed from {url}: {e}")
                continue
        return None
    except Exception as e:
        logger.error(f"All price endpoints failed: {e}")
        return None

def get_features_and_predict(model, symbol="ETHUSDT", threshold=0.73):
    """
    Fetch Binance data, calculate features and predict using the model.
    Returns: current_price, features_array, proba
    """
    try:
        # Try to get full klines data first
        urls = [
            "https://api.binance.com/api/v3/klines",
            "https://api1.binance.com/api/v3/klines",
            "https://api2.binance.com/api/v3/klines",
            "https://api3.binance.com/api/v3/klines"
        ]
        
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 100
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = None
        last_error = None
        
        for url in urls:
            try:
                logger.debug(f"Trying to fetch klines from {url}")
                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                logger.debug(f"Successfully fetched from {url}")
                break
            except Exception as e:
                last_error = e
                logger.debug(f"Failed to fetch from {url}: {e}")
                continue
        
        if response is None:
            logger.warning(f"All klines endpoints failed, last error: {last_error}")
            # Fallback to simple price fetch
            current_price = get_current_price_simple(symbol)
            if current_price is None:
                logger.error("Even simple price fetch failed")
                return None, None, 0.0
            
            # Return with fallback probability
            logger.info(f"Using fallback price: ${current_price:.2f}")
            return current_price, None, 0.65  # Conservative probability when features unavailable
            
        # Parse response data
        try:
            data = response.json()
            if not data or len(data) < 50:  # Need enough data for indicators
                logger.warning(f"Insufficient data received: {len(data) if data else 0} candles")
                current_price = get_current_price_simple(symbol)
                return current_price, None, 0.65 if current_price else (None, None, 0.0)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            current_price = get_current_price_simple(symbol)
            return current_price, None, 0.65 if current_price else (None, None, 0.0)
        
        # Create DataFrame with proper error handling
        try:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric with error handling
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for any NaN values in critical columns
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                logger.warning("NaN values found in price data")
                current_price = get_current_price_simple(symbol)
                return current_price, None, 0.65 if current_price else (None, None, 0.0)
                
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            current_price = get_current_price_simple(symbol)
            return current_price, None, 0.65 if current_price else (None, None, 0.0)
        
        # Calculate technical indicators with error handling
        try:
            # Basic indicators
            df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['EMA_20'] = df['close'].ewm(span=20).mean()
            df['BB_Middle'] = df['SMA_20']
            df['BB_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20, min_periods=1).std()
            df['BB_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20, min_periods=1).std()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9) * 100
            df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
            
            # RSI calculation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_min = df['low'].rolling(window=14, min_periods=1).min()
            high_max = df['high'].rolling(window=14, min_periods=1).max()
            df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean()
            
            # Other indicators
            df['ATR'] = (df['high'] - df['low']).rolling(window=14, min_periods=1).mean()
            df['Volatility'] = (df['high'] - df['low']) / (df['low'] + 1e-9) * 100
            df['Volatility_SMA'] = df['Volatility'].rolling(window=20, min_periods=1).mean()
            df['Volume_Ratio'] = df['volume'] / (df['volume'].rolling(window=20, min_periods=1).mean() + 1e-9)
            
            # Get latest values
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Prepare features array
            features = [
                latest['BB_Upper'], latest['high'], latest['EMA_20'], 
                latest['BB_Middle'], latest['SMA_20'], latest['close'], 
                latest['open'], latest['low'], latest['BB_Lower'],
                latest['RSI'], latest['Stoch_D'], latest['Stoch_K'], 
                latest['BB_Position'], latest['Volatility_SMA'], latest['ATR'], 
                latest['BB_Width'], latest['Volume_Ratio'], latest['Volatility'], 
                latest['volume']
            ]
            
            # Replace NaN values with reasonable defaults
            features = [0.0 if pd.isna(x) or np.isinf(x) else float(x) for x in features]
            features_array = np.array(features).reshape(1, -1)
            
            # Get model prediction
            try:
                if model is not None:
                    proba = float(model.predict_proba(features_array)[0, 1])
                else:
                    proba = 0.5  # Neutral probability when no model
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                proba = 0.5
            
            logger.debug(f"Successfully calculated features for ${current_price:.2f}, probability: {proba:.3f}")
            return current_price, features_array, proba
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            # Try to at least get current price
            try:
                current_price = float(df.iloc[-1]['close'])
                return current_price, None, 0.65
            except:
                current_price = get_current_price_simple(symbol)
                return current_price, None, 0.65 if current_price else (None, None, 0.0)
                
    except Exception as e:
        logger.error(f"Critical error in get_features_and_predict: {e}")
        # Final fallback
        current_price = get_current_price_simple(symbol)
        return current_price, None, 0.65 if current_price else (None, None, 0.0)