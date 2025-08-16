import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

class BinanceDataCollector:
    def __init__(self):
        try:
            # Load environment variables from .env file (find project root)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            env_path = os.path.join(project_root, '.env')
            load_dotenv(env_path)
            
            # Initialize with API keys from .env file
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if api_key and api_secret:
                self.client = Client(api_key, api_secret)
                print(f"✅ Binance client initialized with API key: {api_key[:10]}...")
            else:
                # Use public API without authentication
                self.client = Client()
                print(f"⚠️  Warning: No API keys found, using public API")
                
            self.symbol = 'ETHUSDT'
            self.timeframes = ['1m', '5m', '15m', '30m']
            self.data_dir = "data"
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Binance client initialization error: {e}")
            # Continue without client for testing
            self.client = None
            self.symbol = 'ETHUSDT'
            self.timeframes = ['1m', '5m', '15m', '30m']
            self.data_dir = "data"
            os.makedirs(self.data_dir, exist_ok=True)
    
    def get_recent_data(self, timeframe, minutes=120):
        """
        Get recent data for the specified number of minutes
        This is the main method for real-time trading
        """
        if not hasattr(self, 'client') or self.client is None:
            raise Exception("Binance client not initialized")
            
        try:
            # Calculate how many candles we need based on timeframe
            interval_map = {
                '1m': 1,
                '3m': 3,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            
            candle_minutes = interval_map.get(timeframe, 1)
            limit = min((minutes // candle_minutes) + 1, 1000)  # Binance limit is 1000
            
            # Get klines from Binance
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=timeframe,
                limit=limit
            )
            
            if not klines:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            return df
            
        except Exception as e:
            print(f"Error fetching recent data: {e}")
            raise
    
    def get_historical_data(self, timeframe, days=30):
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        print(f"Collecting {timeframe} data for {self.symbol} from {start_time.date()} to {end_time.date()}")
        
        try:
            klines = self.client.get_historical_klines(
                self.symbol,
                timeframe,
                start_time.strftime("%Y-%m-%d"),
                end_time.strftime("%Y-%m-%d")
            )
            
            if not klines:
                print(f"No data returned for {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.set_index('timestamp', inplace=True)
            
            print(f"✓ Collected {len(df)} records for {timeframe}")
            return df
            
        except Exception as e:
            print(f"✗ Error collecting {timeframe} data: {e}")
            return pd.DataFrame()
    
    def save_data(self, df, timeframe):
        if df.empty:
            return
        
        filename = f"{self.data_dir}/{self.symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename)
        print(f"✓ Saved data to {filename}")
    
    def collect_all_timeframes(self, days=30):
        print(f"Starting data collection for {self.symbol}")
        print(f"Timeframes: {self.timeframes}")
        print(f"Period: {days} days")
        print("-" * 50)
        
        collected_data = {}
        
        for timeframe in self.timeframes:
            df = self.get_historical_data(timeframe, days)
            if not df.empty:
                self.save_data(df, timeframe)
                collected_data[timeframe] = df
            print()
        
        print("Data collection summary:")
        for tf, df in collected_data.items():
            if not df.empty:
                print(f"  {tf}: {len(df)} records ({df.index[0]} to {df.index[-1]})")
        
        return collected_data
    
    def get_latest_price(self):
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting latest price: {e}")
            return None
    
    def get_realtime_data(self, timeframe='1m', limit=100):
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error getting realtime data: {e}")
            return pd.DataFrame()