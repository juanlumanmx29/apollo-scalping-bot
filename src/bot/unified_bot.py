#!/usr/bin/env python3
"""
Unified Trading Bot Base Class
All bots inherit from this class and use config.py for their specific parameters
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import get_bot_config
from src.data_collector.binance_collector import BinanceDataCollector
from src.bot.predictor import PricePredictor

class UnifiedTradingBot:
    def __init__(self, bot_name):
        """Initialize bot with configuration from config.py"""
        # Load bot-specific configuration
        self.bot_name = bot_name.lower()
        self.config = get_bot_config(self.bot_name)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(f"{self.config['name']}")
        
        # Extract configuration parameters
        self.ml_threshold = self.config['ml_threshold']
        self.stop_loss_pct = self.config['stop_loss_pct']
        self.trailing_stop_pct = self.config['trailing_stop_pct']
        self.take_profit_pct = self.config.get('take_profit_pct')
        self.max_loss_pct = self.config['max_loss_pct']
        self.cooldown_minutes = self.config['cooldown_minutes']
        self.min_hold_minutes = self.config['min_hold_minutes']
        self.position_size_eth = self.config['position_size_eth']
        self.fee_rate = self.config['fee_rate']
        self.cycle_sleep = self.config['cycle_sleep_seconds']
        self.min_startup_cycles = self.config['min_startup_cycles']
        
        # Set up wallet paths in centralized wallets folder (absolute path)
        wallet_name = self.config['wallet_name']
        self.wallet_dir = os.path.join(project_root, "wallets")
        self.wallet_file = f"{self.wallet_dir}/{wallet_name}_wallet.json"
        self.report_file = f"{self.wallet_dir}/{wallet_name}_report.md"
        
        # Initialize components
        self.data_collector = BinanceDataCollector()
        model_path = os.path.join(project_root, self.config['model_path'])
        self.predictor = PricePredictor(model_path)
        
        # State tracking
        self.last_trade_time = None
        self.position = None
        self.startup_cycles = 0
        
        # Initialize wallet
        self.init_wallet()
        
        # Log initialization
        self.logger.info(f"🚀 {self.config['name']} Initialized")
        self.logger.info(f"   Description: {self.config['description']}")
        self.logger.info(f"   ML Threshold: {self.ml_threshold:.1%}")
        self.logger.info(f"   Stop Loss: {self.stop_loss_pct:.1%}")
        self.logger.info(f"   Trailing Stop: {self.trailing_stop_pct:.1%}")
        if self.take_profit_pct:
            self.logger.info(f"   Take Profit: {self.take_profit_pct:.1%}")
        else:
            self.logger.info("   Take Profit: None (Let winners run)")
    
    def init_wallet(self):
        """Initialize or load wallet"""
        os.makedirs(self.wallet_dir, exist_ok=True)
        
        if not os.path.exists(self.wallet_file):
            wallet = {
                "balance_usdt": self.config['starting_balance'],
                "balance_eth": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "trades": []
            }
            self.save_wallet(wallet)
    
    def load_wallet(self):
        """Load wallet from file"""
        with open(self.wallet_file, 'r') as f:
            return json.load(f)
    
    def save_wallet(self, wallet):
        """Save wallet to file"""
        wallet['last_updated'] = datetime.now().isoformat()
        with open(self.wallet_file, 'w') as f:
            json.dump(wallet, f, indent=2)
        self.update_report(wallet)
    
    def update_report(self, wallet):
        """Generate markdown report"""
        current_eth_price = 0
        try:
            df = self.data_collector.get_recent_data('1m', minutes=2)
            if df is not None and len(df) > 0:
                current_eth_price = float(df['close'].iloc[-1])
        except:
            pass
        
        eth_value = wallet['balance_eth'] * current_eth_price
        total_value = wallet['balance_usdt'] + eth_value
        total_pnl_pct = (total_value - self.config['starting_balance']) / self.config['starting_balance'] * 100
        
        report = f"""# {self.config['name']} Trading Report

## Balance
- **USDT**: ${wallet['balance_usdt']:.2f}
- **ETH**: {wallet['balance_eth']:.6f} @ ${current_eth_price:.2f}
- **Total Value**: ${total_value:.2f}
- **Total P&L**: ${wallet['total_pnl']:.2f} ({total_pnl_pct:+.2f}%)

## Performance
- **Total Trades**: {wallet['total_trades']}
- **Winning**: {wallet['winning_trades']}
- **Losing**: {wallet['losing_trades']}
- **Win Rate**: {wallet['winning_trades']/wallet['total_trades']*100 if wallet['total_trades'] > 0 else 0:.1f}%
- **Total Fees**: ${wallet.get('total_fees', 0):.2f}

## Recent Trades
| Time | Action | Price | Amount | P&L | Reason |
|------|--------|-------|--------|-----|--------|
"""
        
        # Add last 10 trades
        for trade in wallet['trades'][-10:][::-1]:
            time_str = trade['timestamp'].split('T')[1][:8]
            pnl_str = f"${trade.get('pnl', 0):+.2f}" if 'pnl' in trade else "-"
            report += f"| {time_str} | {trade['action']} | ${trade['price']:.2f} | {trade['amount']:.4f} | {pnl_str} | {trade['reason']} |\n"
        
        # Add position info only if we actually have ETH balance
        if self.position and wallet['balance_eth'] > 0:
            current_pnl_pct = (current_eth_price - self.position['entry_price']) / self.position['entry_price'] * 100
            report += f"\n## Open Position\n"
            report += f"- **Entry**: ${self.position['entry_price']:.2f}\n"
            report += f"- **Amount**: {self.position['amount']:.6f} ETH\n"
            report += f"- **Current P&L**: {current_pnl_pct:+.2f}%\n"
            report += f"- **Stop Loss**: ${self.position['stop_loss']:.2f}\n"
            if self.take_profit_pct:
                report += f"- **Take Profit**: ${self.position['take_profit']:.2f}\n"
        
        report += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        with open(self.report_file, 'w') as f:
            f.write(report)
    
    def get_ml_signal(self, df):
        """Get ML prediction"""
        try:
            should_buy, confidence = self.predictor.predict(df)
            return {
                'signal': 'BUY' if confidence >= self.ml_threshold else 'HOLD',
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}
    
    def calculate_fees(self, amount_usdt):
        """Calculate trading fees"""
        return amount_usdt * self.fee_rate
    
    def execute_buy(self, price, ml_confidence):
        """Execute buy order"""
        wallet = self.load_wallet()
        
        # Check balance
        required_usdt = price * self.position_size_eth * (1 + self.fee_rate)
        if wallet['balance_usdt'] < required_usdt:
            return False, "Insufficient balance"
        
        # Execute trade
        fee = self.calculate_fees(price * self.position_size_eth)
        wallet['balance_usdt'] -= (price * self.position_size_eth + fee)
        wallet['balance_eth'] += self.position_size_eth
        wallet['total_fees'] = wallet.get('total_fees', 0) + fee
        
        # Record position
        # Calculate minimum sell price to cover both fees (0.075% × 2 = 0.15%)
        min_sell_price = price * (1 + self.fee_rate * 2)
        
        self.position = {
            'entry_price': price,
            'entry_time': datetime.now(),
            'amount': self.position_size_eth,
            'max_price': price,
            'ml_confidence': ml_confidence,
            'stop_loss': price * (1 - self.stop_loss_pct),
            'min_sell_price': min_sell_price  # Minimum price to break even after fees
        }
        
        if self.take_profit_pct:
            # Take profit must be above minimum sell price
            self.position['take_profit'] = max(
                price * (1 + self.take_profit_pct),
                min_sell_price * 1.001  # At least 0.1% above break-even
            )
        
        # Record trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'BUY',
            'price': price,
            'amount': self.position_size_eth,
            'fee': fee,
            'reason': f'ML: {ml_confidence:.1%}'
        }
        wallet['trades'].append(trade)
        wallet['total_trades'] += 1
        
        self.save_wallet(wallet)
        self.last_trade_time = datetime.now()
        
        self.logger.info(f"✅ BUY {self.position_size_eth} ETH @ ${price:.2f} (ML: {ml_confidence:.1%})")
        self.logger.info(f"   Min Sell: ${min_sell_price:.2f} | Stop Loss: ${self.position['stop_loss']:.2f}")
        return True, "Buy executed"
    
    def execute_sell(self, price, reason):
        """Execute sell order"""
        if not self.position:
            return False, "No position"
        
        wallet = self.load_wallet()
        
        # Calculate P&L
        fee = self.calculate_fees(price * self.position['amount'])
        gross_pnl = (price - self.position['entry_price']) * self.position['amount']
        net_pnl = gross_pnl - fee - self.calculate_fees(self.position['entry_price'] * self.position['amount'])
        pnl_pct = (price - self.position['entry_price']) / self.position['entry_price'] * 100
        
        # Execute trade
        wallet['balance_usdt'] += (price * self.position['amount'] - fee)
        wallet['balance_eth'] -= self.position['amount']
        wallet['total_pnl'] += net_pnl
        wallet['total_fees'] = wallet.get('total_fees', 0) + fee
        
        if net_pnl > 0:
            wallet['winning_trades'] += 1
        else:
            wallet['losing_trades'] += 1
        
        # Record trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'SELL',
            'price': price,
            'amount': self.position['amount'],
            'fee': fee,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        wallet['trades'].append(trade)
        
        self.save_wallet(wallet)
        
        # Clear position
        self.position = None
        if net_pnl > 0:
            self.last_trade_time = datetime.now()  # Only cooldown on profit
        
        symbol = "🟢" if net_pnl > 0 else "🔴"
        self.logger.info(f"{symbol} SELL @ ${price:.2f} | P&L: {pnl_pct:+.2f}% (${net_pnl:+.2f}) | {reason}")
        return True, "Sell executed"
    
    def check_exit_conditions(self, current_price):
        """Check if we should exit position"""
        if not self.position:
            return False, None
        
        entry_price = self.position['entry_price']
        min_sell_price = self.position.get('min_sell_price', entry_price * 1.0015)
        pnl_pct = (current_price - entry_price) / entry_price
        hold_minutes = (datetime.now() - self.position['entry_time']).total_seconds() / 60
        
        # CRITICAL: Never sell below minimum price (except for max loss protection)
        # This ensures we at least break even after fees
        if current_price < min_sell_price and pnl_pct > -self.max_loss_pct:
            return False, None  # Don't sell at a loss unless it's catastrophic
        
        # Update max price for trailing stop
        if current_price > self.position['max_price']:
            self.position['max_price'] = current_price
        
        # 1. Absolute max loss protection
        if pnl_pct <= -self.max_loss_pct:
            return True, f"Max Loss: {pnl_pct:.2%}"
        
        # 2. Stop loss
        if current_price <= self.position['stop_loss']:
            return True, f"Stop Loss: {pnl_pct:.2%}"
        
        # 3. Take profit (if enabled)
        if self.take_profit_pct and current_price >= self.position['take_profit']:
            return True, f"Take Profit: {pnl_pct:.2%}"
        
        # 4. Trailing stop (only after minimum hold time AND when in profit)
        # IMPORTANT: Trailing stops ONLY protect profits, never cause losses!
        if hold_minutes >= self.min_hold_minutes:
            # Only activate trailing stop if price has gone above entry (we're in profit)
            if self.position['max_price'] > entry_price * 1.001:  # Must be at least 0.1% above entry
                trailing_stop = self.position['max_price'] * (1 - self.trailing_stop_pct)
                # Double-check: trailing stop must be above entry price
                if trailing_stop > entry_price and current_price <= trailing_stop:
                    return True, f"Trailing Stop: {pnl_pct:.2%}"
        
        return False, None
    
    def is_in_cooldown(self):
        """Check if we're in cooldown period"""
        if not self.last_trade_time:
            return False
        
        time_since_trade = (datetime.now() - self.last_trade_time).total_seconds() / 60
        return time_since_trade < self.cooldown_minutes
    
    def run_cycle(self):
        """Run one trading cycle"""
        try:
            # Get market data
            df = self.data_collector.get_recent_data('1m', minutes=120)
            if df is None or len(df) < 60:
                return
            
            current_price = float(df['close'].iloc[-1])
            
            # Startup observation period
            if self.startup_cycles < self.min_startup_cycles:
                self.startup_cycles += 1
                self.logger.info(f"📊 Startup {self.startup_cycles}/{self.min_startup_cycles} | Price: ${current_price:.2f}")
                return
            
            # Check if we have a position
            if self.position:
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(current_price)
                if should_exit:
                    self.execute_sell(current_price, exit_reason)
                else:
                    pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price'] * 100
                    self.logger.info(f"📊 HOLDING | ${current_price:.2f} | P&L: {pnl_pct:+.2f}%")
            else:
                # Check for entry
                if self.is_in_cooldown():
                    remaining = self.cooldown_minutes - (datetime.now() - self.last_trade_time).total_seconds() / 60
                    self.logger.info(f"⏸️  Cooldown: {remaining:.1f}min | Price: ${current_price:.2f}")
                    return
                
                # Get ML signal
                ml_signal = self.get_ml_signal(df)
                
                if ml_signal['signal'] == 'BUY':
                    self.execute_buy(current_price, ml_signal['confidence'])
                else:
                    self.logger.info(f"📊 WAITING | ${current_price:.2f} | ML: {ml_signal['confidence']:.1%}")
            
        except Exception as e:
            self.logger.error(f"Cycle error: {e}")
    
    def run(self):
        """Main loop"""
        self.logger.info(f"🚀 Starting {self.config['name']} Trading Bot")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(self.cycle_sleep)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️  Bot stopped by user")
            if self.position:
                self.logger.warning("⚠️  Warning: Open position remains")