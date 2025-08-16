#!/usr/bin/env python3

"""
Momentum Trading Bot - Follow the trend, don't predict it
Uses price momentum and volume to ride trends instead of predicting reversals
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_collector.binance_collector import BinanceDataCollector
from src.wallet.paper_wallet import PaperWallet

class MomentumBot:
    def __init__(self, 
                 momentum_threshold=0.002,      # 0.2% momentum to enter
                 volume_multiplier=1.5,         # 1.5x average volume required
                 stop_loss_pct=0.005,           # 0.5% stop loss (wider)
                 trailing_stop_pct=0.003,       # 0.3% trailing stop
                 take_profit_pct=0.01,          # 1.0% take profit
                 cooldown_minutes=20,           # 20 min cooldown
                 position_size_eth=1.3):        # 1.3 ETH per trade
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.data_collector = BinanceDataCollector()
        self.wallet = PaperWallet("wallet/momentum_wallet.json", "wallet/momentum_wallet.md")
        
        # Trading parameters
        self.momentum_threshold = momentum_threshold
        self.volume_multiplier = volume_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.cooldown_minutes = cooldown_minutes
        self.position_size_eth = position_size_eth
        
        # State
        self.last_profitable_trade_time = None
        
        self.logger.info("🚀 Momentum Bot Initialized")
        self.logger.info(f"   Momentum Threshold: {momentum_threshold:.2%}")
        self.logger.info(f"   Volume Multiplier: {volume_multiplier}x")
        self.logger.info(f"   Stop Loss: {stop_loss_pct:.1%}")
        self.logger.info(f"   Take Profit: {take_profit_pct:.1%}")
    
    def calculate_momentum(self, df, periods=5):
        """Calculate price momentum over last N periods"""
        if len(df) < periods + 1:
            return 0
        
        # Price momentum (rate of change)
        current_price = float(df['close'].iloc[-1])
        past_price = float(df['close'].iloc[-periods-1])
        momentum = (current_price - past_price) / past_price
        
        return momentum
    
    def check_volume_surge(self, df, lookback=20):
        """Check if current volume is surging"""
        if len(df) < lookback + 1:
            return False
        
        # Current volume
        current_volume = float(df['volume'].iloc[-1])
        
        # Average volume over lookback period
        avg_volume = float(df['volume'].iloc[-lookback-1:-1].mean())
        
        # Check if current volume exceeds threshold
        return current_volume > (avg_volume * self.volume_multiplier)
    
    def detect_breakout(self, df):
        """Detect momentum breakout with volume confirmation"""
        # Calculate momentum
        momentum = self.calculate_momentum(df, periods=5)
        
        # Check volume surge
        volume_surge = self.check_volume_surge(df, lookback=20)
        
        # Breakout conditions
        bullish_breakout = momentum > self.momentum_threshold and volume_surge
        bearish_breakout = momentum < -self.momentum_threshold and volume_surge
        
        return {
            'bullish': bullish_breakout,
            'bearish': bearish_breakout,
            'momentum': momentum,
            'volume_surge': volume_surge
        }
    
    def is_in_cooldown(self):
        """Check if we're in cooldown period"""
        if self.last_profitable_trade_time is None:
            return False
        
        time_diff = (datetime.now() - self.last_profitable_trade_time).total_seconds() / 60
        return time_diff < self.cooldown_minutes
    
    def should_buy(self, df, current_price):
        """Check for bullish momentum breakout"""
        # Don't buy if we have a position
        if self.wallet.get_open_positions():
            return False, "Already have position"
        
        # Check cooldown
        if self.is_in_cooldown():
            remaining = self.cooldown_minutes - (datetime.now() - self.last_profitable_trade_time).total_seconds() / 60
            return False, f"Cooldown: {remaining:.1f}min remaining"
        
        # Check wallet balance
        balance = self.wallet.get_balance('USDT')
        required = current_price * self.position_size_eth * 1.001
        if balance < required:
            return False, f"Insufficient funds: ${balance:.2f} < ${required:.2f}"
        
        # Detect breakout
        breakout = self.detect_breakout(df)
        
        if breakout['bullish']:
            return True, f"Bullish momentum: {breakout['momentum']:.2%} + Volume surge"
        else:
            return False, f"No breakout (Momentum: {breakout['momentum']:.2%})"
    
    def should_sell(self, df, current_price):
        """Check sell conditions"""
        open_positions = self.wallet.get_open_positions()
        if not open_positions:
            return False, None, "No position"
        
        # Get position
        position_id, position = next(iter(open_positions.items()))
        entry_price = position['entry_price']
        max_price = position['max_price_seen']
        
        # Update max price
        if current_price > max_price:
            self.wallet.update_position_max_price(position_id, current_price)
            max_price = current_price
            self.logger.info(f"📈 New MAX: ${max_price:.2f} (Entry: ${entry_price:.2f}, +{((max_price/entry_price - 1)):.2%})")
        
        # Calculate P&L
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return True, position_id, f"Take Profit: {pnl_pct:.2%}"
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, position_id, f"Stop Loss: {pnl_pct:.2%}"
        
        # Trailing stop (only if profitable)
        if current_price > entry_price * 1.0015:  # Above fees
            trailing_stop = max_price * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop:
                return True, position_id, f"Trailing Stop: {pnl_pct:.2%}"
        
        # Check for momentum reversal
        breakout = self.detect_breakout(df)
        if breakout['bearish'] and pnl_pct > 0:
            return True, position_id, f"Momentum reversal: {pnl_pct:.2%}"
        
        return False, position_id, f"Holding: {pnl_pct:.2%}"
    
    def execute_buy(self, current_price, reason):
        """Execute buy order"""
        fee_rate = 0.00075
        usdt_amount = (current_price * self.position_size_eth) / (1 - fee_rate)
        fee_amount = usdt_amount * fee_rate
        
        success = self.wallet.buy_eth(usdt_amount, current_price, 0, reason)
        if success:
            dt = datetime.now().strftime('%H:%M:%S')
            self.logger.info(f"✅ BUY  | {dt} | {self.position_size_eth} ETH | ${current_price:.2f} | Fee: ${fee_amount:.2f} | {reason}")
            return True
        return False
    
    def execute_sell(self, position_id, current_price, reason):
        """Execute sell order"""
        open_positions = self.wallet.get_open_positions()
        position = open_positions.get(position_id)
        
        if position:
            entry_price = position['entry_price']
            amount_eth = position['amount_eth']
            pnl_pct = (current_price - entry_price) / entry_price
            pnl_usd = (current_price - entry_price) * amount_eth
        
        success = self.wallet.sell_eth(position_id, current_price, reason)
        if success:
            dt = datetime.now().strftime('%H:%M:%S')
            self.logger.info(f"✅ SELL | {dt} | {amount_eth} ETH | ${current_price:.2f} | P&L: {pnl_pct:+.2%} (${pnl_usd:+.2f}) | {reason}")
            
            # Start cooldown for profitable trades
            if pnl_pct > 0:
                self.last_profitable_trade_time = datetime.now()
                self.logger.info(f"🧊 Cooldown: {self.cooldown_minutes} minutes")
            
            return True
        return False
    
    def run_cycle(self):
        """Run one trading cycle"""
        try:
            # Get market data
            df = self.data_collector.get_recent_data('1m', minutes=60)
            if df is None or len(df) < 30:
                self.logger.warning("Insufficient market data")
                return
            
            current_price = float(df['close'].iloc[-1])
            
            # Check for sell signals first
            should_sell, position_id, sell_reason = self.should_sell(df, current_price)
            
            if should_sell:
                self.execute_sell(position_id, current_price, sell_reason)
            else:
                # Check for buy signals
                should_buy, buy_reason = self.should_buy(df, current_price)
                
                if should_buy:
                    self.execute_buy(current_price, buy_reason)
                else:
                    # Log status
                    dt = datetime.now().strftime('%H:%M:%S')
                    open_positions = self.wallet.get_open_positions()
                    
                    if open_positions:
                        for pos_id, pos in open_positions.items():
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                            pnl_usd = (current_price - pos['entry_price']) * pos['amount_eth']
                            self.logger.info(f"📊 {dt} | ${current_price:.2f} | P&L: {pnl_pct:+.3%} (${pnl_usd:+.2f}) | {sell_reason}")
                    else:
                        breakout = self.detect_breakout(df)
                        self.logger.info(f"📊 {dt} | ${current_price:.2f} | No Position | Momentum: {breakout['momentum']:+.3%} | {buy_reason}")
        
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def run_continuous(self, cycle_seconds=30):
        """Run bot continuously"""
        self.logger.info(f"🚀 Starting Momentum Bot - {cycle_seconds}s cycles")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(cycle_seconds)
        except KeyboardInterrupt:
            self.logger.info("⏹️  Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")

def main():
    # Conservative momentum trading
    bot = MomentumBot(
        momentum_threshold=0.003,      # 0.3% momentum required
        volume_multiplier=2.0,         # 2x volume spike required
        stop_loss_pct=0.005,           # 0.5% stop loss
        trailing_stop_pct=0.003,       # 0.3% trailing stop
        take_profit_pct=0.01,          # 1.0% take profit
        cooldown_minutes=20,           # 20 min cooldown
        position_size_eth=1.3          # 1.3 ETH positions
    )
    
    bot.run_continuous(cycle_seconds=10)

if __name__ == "__main__":
    main()