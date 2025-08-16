#!/usr/bin/env python3
"""
Wallet Viewer - View all bot wallets in one place
Shows current balances, P&L, and recent trades
"""

import json
import os
from datetime import datetime

def load_wallet(wallet_path):
    """Load wallet data from JSON file"""
    if not os.path.exists(wallet_path):
        return None
    
    try:
        with open(wallet_path, 'r') as f:
            return json.load(f)
    except:
        return None

def format_currency(amount):
    """Format currency with proper commas and decimals"""
    return f"${amount:,.2f}"

def format_percentage(percentage):
    """Format percentage with proper sign"""
    return f"{percentage:+.2f}%"

def get_current_eth_value(wallet):
    """Get current ETH value from last trade or recent market price"""
    eth_balance = wallet.get('balance_eth', 0)
    
    # Get the most recent ETH price from trades
    trades = wallet.get('trades', [])
    if trades:
        # Look for the last trade price
        last_trade = trades[-1]
        estimated_eth_price = last_trade.get('price', 4600)
    else:
        estimated_eth_price = 4600  # Default fallback
    
    return eth_balance * estimated_eth_price

def display_wallet_summary():
    """Display summary of all wallets"""
    print("=" * 80)
    print("🚀 APOLLO TRADING BOTS - WALLET SUMMARY")
    print("=" * 80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    wallets = {
        'apollo': 'Apollo (Scalping)',
        'midas': 'Midas (Wild)',
        'athena': 'Athena (Controlled Wild)'
    }
    
    total_usdt = 0
    total_eth = 0
    total_pnl = 0
    total_trades = 0
    
    for bot_name, display_name in wallets.items():
        wallet_path = f"wallets/{bot_name}_wallet.json"
        wallet = load_wallet(wallet_path)
        
        if wallet is None:
            print(f"❌ {display_name}: No wallet found")
            continue
        
        # Extract data
        usdt_balance = wallet.get('balance_usdt', 0)
        eth_balance = wallet.get('balance_eth', 0)
        pnl = wallet.get('total_pnl', 0)
        trades = wallet.get('total_trades', 0)
        winning_trades = wallet.get('winning_trades', 0)
        losing_trades = wallet.get('losing_trades', 0)
        
        # Calculate win rate
        win_rate = (winning_trades / trades * 100) if trades > 0 else 0
        
        # Estimate total value
        eth_value = get_current_eth_value(wallet)
        total_value = usdt_balance + eth_value
        starting_balance = 5000
        total_pnl_pct = (total_value - starting_balance) / starting_balance * 100
        
        # Add to totals
        total_usdt += usdt_balance
        total_eth += eth_balance
        total_pnl += pnl
        total_trades += trades
        
        print(f"📊 {display_name}")
        print(f"   Balance: {format_currency(usdt_balance)} USDT + {eth_balance:.4f} ETH")
        print(f"   Est. Value: {format_currency(total_value)} ({format_percentage(total_pnl_pct)})")
        print(f"   P&L: {format_currency(pnl)} | Trades: {trades} | Win Rate: {win_rate:.1f}%")
        
        # Show recent trades
        recent_trades = wallet.get('trades', [])[-3:]  # Last 3 trades
        if recent_trades:
            print(f"   Recent Trades:")
            for trade in recent_trades[-2:]:  # Show last 2
                time_str = trade.get('timestamp', '').split('T')[1][:8]
                action = trade.get('action', '')
                price = trade.get('price', 0)
                pnl = trade.get('pnl', 0)
                reason = trade.get('reason', '')
                
                if action == 'BUY':
                    print(f"     {time_str} BUY @ {format_currency(price)} - {reason}")
                else:
                    pnl_str = f"({format_currency(pnl)})" if pnl != 0 else ""
                    print(f"     {time_str} SELL @ {format_currency(price)} {pnl_str} - {reason}")
        print()
    
    # Overall summary
    print("=" * 40)
    print(f"💰 TOTAL PORTFOLIO")
    print(f"   USDT: {format_currency(total_usdt)}")
    print(f"   ETH: {total_eth:.4f}")
    print(f"   Total P&L: {format_currency(total_pnl)}")
    print(f"   Total Trades: {total_trades}")
    print("=" * 40)

def display_detailed_wallet(bot_name):
    """Display detailed view of a specific wallet"""
    wallet_path = f"wallets/{bot_name}_wallet.json"
    wallet = load_wallet(wallet_path)
    
    if wallet is None:
        print(f"❌ No wallet found for {bot_name}")
        return
    
    print(f"📊 DETAILED WALLET: {bot_name.upper()}")
    print("=" * 50)
    
    # All trades
    trades = wallet.get('trades', [])
    print(f"All Trades ({len(trades)}):")
    
    for i, trade in enumerate(trades, 1):
        timestamp = trade.get('timestamp', '').replace('T', ' ')[:19]
        action = trade.get('action', '')
        price = trade.get('price', 0)
        amount = trade.get('amount', 0)
        fee = trade.get('fee', 0)
        pnl = trade.get('pnl', 0)
        reason = trade.get('reason', '')
        
        if action == 'BUY':
            print(f"{i:2d}. {timestamp} BUY  {amount:.4f} ETH @ {format_currency(price)} (Fee: {format_currency(fee)}) - {reason}")
        else:
            pnl_str = f"P&L: {format_currency(pnl)}, " if pnl != 0 else ""
            print(f"{i:2d}. {timestamp} SELL {amount:.4f} ETH @ {format_currency(price)} ({pnl_str}Fee: {format_currency(fee)}) - {reason}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2:
        display_detailed_wallet(sys.argv[1])
    else:
        display_wallet_summary()
        print()
        print("💡 Usage:")
        print("   python view_wallets.py           # View summary")
        print("   python view_wallets.py apollo    # View Apollo details")
        print("   python view_wallets.py midas     # View Midas details") 
        print("   python view_wallets.py athena    # View Athena details")