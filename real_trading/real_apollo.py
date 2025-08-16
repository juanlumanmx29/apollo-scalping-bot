#!/usr/bin/env python3
"""
Real Apollo Bot - Same as Apollo but with REAL Binance trading
Uses the configuration from .env file in this directory
"""

import os
import sys
import time
import json
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

# Add parent directory to path to import Apollo components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from local .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Import Apollo components
from src.bot.unified_bot import ApolloBot
from binance.client import Client
from binance.exceptions import BinanceAPIException

class RealApollo(ApolloBot):
    """Real Apollo - Same as Apollo but trades with real money"""
    
    def __init__(self):
        # Initialize Apollo with all its standard settings
        super().__init__(bot_name="apollo")
        
        print("🚨 REAL APOLLO STARTING - LIVE TRADING MODE 🚨")
        print("=" * 50)
        
        # Real trading configuration from .env
        self.trading_mode = os.getenv('TRADING_MODE', 'REAL')
        self.min_bnb_balance = float(os.getenv('MIN_BNB_BALANCE', 0.1))
        self.bnb_refill_amount = float(os.getenv('BNB_REFILL_AMOUNT', 1.0))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE_USDT', 1000))
        self.trading_amount = float(os.getenv('TRADING_USDT_AMOUNT', 100))
        
        # Initialize real Binance client
        self.real_client = self._init_real_binance_client()
        
        # Override wallet file path for real trading
        self.wallet_file = os.path.join(os.path.dirname(__file__), 'real_apollo_wallet.json')
        
        # Initialize real wallet
        self.real_wallet = self._load_real_wallet()
        
        print(f"✅ Real Apollo initialized")
        print(f"💰 Trading Amount: ${self.trading_amount}")
        print(f"🎯 Max Position: ${self.max_position_size}")
        print(f"🔄 Trading Mode: {self.trading_mode}")
        print("=" * 50)
    
    def _init_real_binance_client(self):
        """Initialize real Binance client with API credentials"""
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("❌ Binance API credentials not found in .env file")
        
        client = Client(api_key, secret_key)
        
        # Test connection
        try:
            account = client.get_account()
            print(f"✅ Connected to Binance account")
            return client
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Binance: {e}")
    
    def _load_real_wallet(self):
        """Load or create real trading wallet"""
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    wallet = json.load(f)
                print(f"💰 Loaded real wallet: {wallet.get('total_trades', 0)} trades")
                return wallet
            except Exception as e:
                print(f"⚠️ Error loading wallet: {e}")
        
        # Create new real wallet
        wallet = {
            "balance_usdt": 1000.0,
            "balance_eth": 0.0,
            "balance_bnb": 0.0,
            "total_value_usdt": 1000.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_fees_bnb": 0.0,
            "trades": [],
            "last_updated": datetime.now().isoformat()
        }
        self._save_real_wallet(wallet)
        return wallet
    
    def _save_real_wallet(self, wallet=None):
        """Save real wallet to file"""
        if wallet is None:
            wallet = self.real_wallet
        
        wallet['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.wallet_file, 'w') as f:
                json.dump(wallet, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving real wallet: {e}")
    
    def _adjust_quantity_for_binance(self, quantity):
        """Adjust quantity to meet Binance lot size requirements"""
        try:
            # Get symbol info for ETHUSDT
            symbol_info = self.real_client.get_symbol_info('ETHUSDT')
            
            # Find LOT_SIZE filter
            lot_size_filter = None
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    lot_size_filter = f
                    break
            
            if lot_size_filter:
                min_qty = float(lot_size_filter['minQty'])
                step_size = float(lot_size_filter['stepSize'])
                
                # Adjust to step size
                qty_decimal = Decimal(str(quantity))
                step_decimal = Decimal(str(step_size))
                
                # Round down to nearest step
                adjusted = (qty_decimal // step_decimal) * step_decimal
                
                # Ensure minimum quantity
                if adjusted < Decimal(str(min_qty)):
                    adjusted = Decimal(str(min_qty))
                
                return float(adjusted)
            
        except Exception as e:
            print(f"⚠️ Error adjusting quantity: {e}")
        
        return quantity
    
    def execute_real_buy(self, price, confidence):
        """Execute real buy order on Binance"""
        try:
            # Calculate quantity based on trading amount
            quantity = self.trading_amount / price
            quantity = self._adjust_quantity_for_binance(quantity)
            
            print(f"🔥 EXECUTING REAL BUY ORDER")
            print(f"💰 Quantity: {quantity:.8f} ETH @ ${price:.2f}")
            print(f"🤖 ML Confidence: {confidence:.1f}%")
            
            # Execute market buy order
            order = self.real_client.order_market_buy(
                symbol='ETHUSDT',
                quantity=quantity
            )
            
            print(f"✅ REAL BUY ORDER EXECUTED - ID: {order['orderId']}")
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY',
                'price': price,
                'amount': quantity,
                'reason': f'ML: {confidence:.1f}%',
                'order_id': order['orderId'],
                'real_order': True
            }
            
            self.real_wallet['trades'].append(trade)
            self.real_wallet['total_trades'] += 1
            self._save_real_wallet()
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing real buy: {e}")
            return False
    
    def execute_real_sell(self, price, reason):
        """Execute real sell order on Binance"""
        try:
            # Get current ETH balance
            account = self.real_client.get_account()
            eth_balance = 0.0
            
            for balance in account['balances']:
                if balance['asset'] == 'ETH':
                    eth_balance = float(balance['free'])
                    break
            
            if eth_balance < 0.001:  # Less than 0.001 ETH
                print("⚠️ No ETH balance to sell")
                return False
            
            # Adjust quantity for Binance requirements
            quantity = self._adjust_quantity_for_binance(eth_balance)
            
            print(f"🔥 EXECUTING REAL SELL ORDER")
            print(f"💰 Quantity: {quantity:.8f} ETH @ ${price:.2f}")
            print(f"📝 Reason: {reason}")
            
            # Execute market sell order
            order = self.real_client.order_market_sell(
                symbol='ETHUSDT',
                quantity=quantity
            )
            
            print(f"✅ REAL SELL ORDER EXECUTED - ID: {order['orderId']}")
            
            # Calculate P&L if we can find the buy trade
            pnl = 0.0
            pnl_pct = 0.0
            
            # Find the corresponding buy trade
            trades = self.real_wallet.get('trades', [])
            for i in range(len(trades) - 1, -1, -1):
                if trades[i]['action'] == 'BUY' and 'pnl' not in trades[i]:
                    buy_price = trades[i]['price']
                    pnl = (price - buy_price) * quantity
                    pnl_pct = (price - buy_price) / buy_price
                    break
            
            # Record the trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'price': price,
                'amount': quantity,
                'reason': reason,
                'order_id': order['orderId'],
                'real_order': True,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            self.real_wallet['trades'].append(trade)
            self.real_wallet['total_trades'] += 1
            
            if pnl > 0:
                self.real_wallet['winning_trades'] += 1
            else:
                self.real_wallet['losing_trades'] += 1
            
            self.real_wallet['total_pnl'] += pnl
            self._save_real_wallet()
            
            print(f"💰 Trade P&L: ${pnl:.2f} ({pnl_pct*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error executing real sell: {e}")
            return False
    
    def buy(self, price, confidence):
        """Override buy method to use real trading"""
        # First do the regular Apollo buy logic (for position tracking, etc.)
        super().buy(price, confidence)
        
        # Then execute the real trade
        return self.execute_real_buy(price, confidence)
    
    def sell(self, price, reason):
        """Override sell method to use real trading"""
        # First do the regular Apollo sell logic (for position tracking, etc.)
        super().sell(price, reason)
        
        # Then execute the real trade
        return self.execute_real_sell(price, reason)
    
    def run(self):
        """Run Real Apollo with all Apollo's logic but real trading"""
        print("🚀 Starting Real Apollo Bot...")
        print("⚠️  This bot trades with REAL MONEY!")
        
        # Run the standard Apollo bot logic
        # This includes all the ML predictions, timing, strategy, etc.
        super().run()

def main():
    """Main entry point for Real Apollo"""
    print("=" * 60)
    print("🚀 REAL APOLLO TRADING BOT")
    print("💰 TRADING WITH REAL MONEY ON BINANCE")
    print("⚠️  MAKE SURE YOU UNDERSTAND THE RISKS")
    print("=" * 60)
    
    try:
        real_apollo = RealApollo()
        real_apollo.run()
    except KeyboardInterrupt:
        print("\n🛑 Real Apollo stopped by user")
    except Exception as e:
        print(f"❌ Fatal error in Real Apollo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())