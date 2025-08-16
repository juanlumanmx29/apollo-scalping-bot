#!/usr/bin/env python3
"""
Model Backtesting System
Evaluate ML models against historical data to see actual performance
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from src.data_collector.binance_collector import BinanceDataCollector
from src.model.feature_engineer import FeatureEngineer
from src.bot.predictor import PricePredictor

class ModelBacktester:
    def __init__(self, model_path, start_balance=5000):
        self.model_path = model_path
        self.start_balance = start_balance
        self.predictor = PricePredictor(model_path)
        self.data_collector = BinanceDataCollector()
        
    def backtest(self, days_back=7, ml_threshold=0.7):
        """Backtest model on historical data"""
        print(f"🔍 Backtesting model: {os.path.basename(self.model_path)}")
        print(f"📊 Parameters: {days_back} days, {ml_threshold:.1%} threshold")
        
        # Get historical data
        df = self.data_collector.get_recent_data('1m', minutes=days_back * 24 * 60)
        if df is None or len(df) < 100:
            print("❌ Insufficient historical data")
            return None
        
        print(f"📈 Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        # Simulation variables
        balance_usdt = self.start_balance
        balance_eth = 0.0
        position = None
        trades = []
        
        fee_rate = 0.00075  # 0.075% Binance fee
        
        # Walk through each candle
        for i in range(100, len(df)):  # Start at 100 to have history for features
            current_data = df.iloc[:i+1]
            current_price = float(df.iloc[i]['close'])
            current_time = df.index[i]
            
            # Get ML prediction
            try:
                result = self.predictor.predict(current_data)
                if not result:
                    continue
                    
                should_buy, confidence = result
                confidence = float(confidence) if confidence else 0.0
                
            except Exception as e:
                continue
            
            # Trading logic
            if position is None and confidence >= ml_threshold:
                # BUY
                if balance_usdt > current_price * 1.001:  # Have enough for 1+ ETH
                    amount_eth = balance_usdt / (current_price * (1 + fee_rate))
                    fee = balance_usdt * fee_rate
                    balance_usdt = 0
                    balance_eth = amount_eth
                    
                    position = {
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'amount_eth': amount_eth,
                        'confidence': confidence
                    }
                    
                    trades.append({
                        'time': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'amount_eth': amount_eth,
                        'confidence': confidence,
                        'fee': fee
                    })
            
            elif position is not None:
                # Check exit conditions
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                should_sell = False
                sell_reason = ""
                
                # Stop loss: -0.3%
                if pnl_pct <= -0.003:
                    should_sell = True
                    sell_reason = "Stop Loss"
                
                # Take profit: varies by confidence
                elif confidence >= 0.8 and pnl_pct >= 0.006:  # High confidence: 0.6%
                    should_sell = True
                    sell_reason = "Take Profit (High Conf)"
                elif pnl_pct >= 0.004:  # Normal: 0.4%
                    should_sell = True
                    sell_reason = "Take Profit"
                
                if should_sell:
                    # SELL
                    gross_usdt = current_price * balance_eth
                    fee = gross_usdt * fee_rate
                    balance_usdt = gross_usdt - fee
                    
                    profit_loss = balance_usdt - self.start_balance
                    
                    trades.append({
                        'time': current_time,
                        'action': 'SELL',
                        'price': current_price,
                        'amount_eth': balance_eth,
                        'pnl_pct': pnl_pct,
                        'profit_loss': profit_loss,
                        'reason': sell_reason,
                        'fee': fee,
                        'duration_minutes': (current_time - position['entry_time']).total_seconds() / 60
                    })
                    
                    balance_eth = 0
                    position = None
        
        # Final position value
        if position is not None:
            final_price = float(df.iloc[-1]['close'])
            final_value = final_price * balance_eth
        else:
            final_value = balance_usdt
        
        return self.analyze_results(trades, final_value)
    
    def analyze_results(self, trades, final_value):
        """Analyze backtest results"""
        if not trades:
            print("❌ No trades executed")
            return None
        
        # Separate buy/sell trades
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']
        
        total_return = ((final_value - self.start_balance) / self.start_balance) * 100
        
        if sells:
            profits = [t['profit_loss'] for t in sells]
            win_trades = [p for p in profits if p > 0]
            loss_trades = [p for p in profits if p <= 0]
            
            win_rate = len(win_trades) / len(sells) * 100 if sells else 0
            avg_win = np.mean(win_trades) if win_trades else 0
            avg_loss = np.mean(loss_trades) if loss_trades else 0
            total_fees = sum(t.get('fee', 0) for t in trades)
            
            avg_duration = np.mean([t.get('duration_minutes', 0) for t in sells])
            
            # Get confidence stats
            confidences = [t['confidence'] for t in buys]
            
            results = {
                'total_trades': len(buys),
                'completed_trades': len(sells),
                'win_rate': win_rate,
                'total_return': total_return,
                'final_value': final_value,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_fees': total_fees,
                'avg_duration': avg_duration,
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'trades': trades
            }
            
            self.print_results(results)
            return results
        else:
            print(f"⚠️  {len(buys)} positions opened but none closed")
            return None
    
    def print_results(self, results):
        """Print backtest results"""
        print("\n" + "="*50)
        print("📊 BACKTEST RESULTS")
        print("="*50)
        print(f"💰 Final Value: ${results['final_value']:,.2f}")
        print(f"📈 Total Return: {results['total_return']:+.2f}%")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"📊 Trades: {results['completed_trades']} completed / {results['total_trades']} total")
        print(f"💚 Avg Win: ${results['avg_win']:+.2f}")
        print(f"💔 Avg Loss: ${results['avg_loss']:+.2f}")
        print(f"💸 Total Fees: ${results['total_fees']:.2f}")
        print(f"⏱️  Avg Duration: {results['avg_duration']:.1f} minutes")
        print(f"🧠 ML Confidence: {results['avg_confidence']:.1%} (range: {results['min_confidence']:.1%}-{results['max_confidence']:.1%})")
        print()
        
        if results['total_return'] > 0:
            print("✅ Model shows PROFIT potential")
        else:
            print("❌ Model shows LOSS - needs improvement")

def main():
    """Test both models with different thresholds"""
    models = [
        ('models/apollo_model_latest.joblib', 'Apollo Model'),
        ('models/scalping_model_5min.joblib', 'Scalping Model')
    ]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for model_path, model_name in models:
        if os.path.exists(model_path):
            print(f"\n🚀 Testing {model_name}")
            print("-" * 40)
            
            for threshold in thresholds:
                print(f"\n📊 Threshold: {threshold:.1%}")
                tester = ModelBacktester(model_path)
                tester.backtest(days_back=3, ml_threshold=threshold)
        else:
            print(f"❌ Model not found: {model_path}")

if __name__ == "__main__":
    main()