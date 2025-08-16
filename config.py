#!/usr/bin/env python3
"""
Bot Configuration File
Centralized configuration for all trading bots
"""

# Global Settings
STARTING_BALANCE = 5000.0
POSITION_SIZE_ETH = 1.0
FEE_RATE = 0.00075  # 0.075% Binance fee
CYCLE_SLEEP_SECONDS = 10
MIN_STARTUP_CYCLES = 3

# Model Settings
MODEL_PATH = "models/smart_entry_model_20250810_172020.joblib"  # Latest smart entry model
ML_THRESHOLD_DEFAULT = 0.70  # Default ML confidence threshold

# Bot-Specific Configurations
BOT_CONFIGS = {
    "apollo": {
        "name": "Apollo",
        "description": "True Scalping Bot - Small gains and losses",
        "ml_threshold": 0.70,           # 70% ML confidence
        "stop_loss_pct": 0.01,          # 1% stop loss
        "trailing_stop_pct": 0.002,     # 0.2% trailing stop
        "take_profit_pct": 0.003,       # 0.3% take profit
        "max_loss_pct": 0.015,          # 1.5% absolute max loss
        "cooldown_minutes": 2,          # 2 minute cooldown after profitable trades
        "min_hold_minutes": 3,          # 3 minute minimum hold
        "folder": "bots/apollo",
        "wallet_name": "apollo"
    },
    
    "midas": {
        "name": "Midas",
        "description": "Wild Bot - 5% stop loss, NO take profit, let winners run",
        "ml_threshold": 0.80,           # 80% ML confidence (more conservative entry)
        "stop_loss_pct": 0.05,          # 5% stop loss
        "trailing_stop_pct": 0.006,     # 0.6% trailing stop
        "take_profit_pct": None,        # NO take profit - let winners run
        "max_loss_pct": 0.06,           # 6% absolute max loss
        "cooldown_minutes": 15,         # 15 minute cooldown
        "min_hold_minutes": 30,         # 30 minute minimum hold
        "override_threshold": 0.90,     # 90% confidence for override
        "folder": "bots/midas",
        "wallet_name": "midas"
    },
    
    "athena": {
        "name": "Athena",
        "description": "Like Midas but with 2% take profit for controlled exits",
        "ml_threshold": 0.80,           # 80% ML confidence
        "stop_loss_pct": 0.05,          # 5% stop loss
        "trailing_stop_pct": 0.006,     # 0.6% trailing stop
        "take_profit_pct": 0.02,        # 2% take profit
        "max_loss_pct": 0.06,           # 6% absolute max loss
        "cooldown_minutes": 15,         # 15 minute cooldown
        "min_hold_minutes": 30,         # 30 minute minimum hold
        "override_threshold": 0.90,     # 90% confidence for override
        "folder": "bots/athena",
        "wallet_name": "athena"
    }
}

def get_bot_config(bot_name):
    """Get configuration for a specific bot"""
    if bot_name.lower() not in BOT_CONFIGS:
        raise ValueError(f"Bot '{bot_name}' not found in configurations")
    
    config = BOT_CONFIGS[bot_name.lower()].copy()
    
    # Add global settings to bot config
    config.update({
        "starting_balance": STARTING_BALANCE,
        "position_size_eth": POSITION_SIZE_ETH,
        "fee_rate": FEE_RATE,
        "cycle_sleep_seconds": CYCLE_SLEEP_SECONDS,
        "min_startup_cycles": MIN_STARTUP_CYCLES,
        "model_path": MODEL_PATH
    })
    
    return config

def print_bot_configs():
    """Print all bot configurations for reference"""
    print("\n🤖 Bot Configurations:")
    print("=" * 50)
    
    for bot_name, config in BOT_CONFIGS.items():
        print(f"\n{config['name']} ({bot_name.upper()})")
        print(f"Description: {config['description']}")
        print(f"ML Threshold: {config['ml_threshold']:.1%}")
        print(f"Stop Loss: {config['stop_loss_pct']:.1%}")
        print(f"Trailing Stop: {config['trailing_stop_pct']:.1%}")
        
        if config['take_profit_pct']:
            print(f"Take Profit: {config['take_profit_pct']:.1%}")
        else:
            print("Take Profit: None (Let winners run)")
            
        print(f"Max Loss: {config['max_loss_pct']:.1%}")
        print(f"Cooldown: {config['cooldown_minutes']} minutes")
        print(f"Min Hold: {config['min_hold_minutes']} minutes")

if __name__ == "__main__":
    print_bot_configs()