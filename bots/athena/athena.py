#!/usr/bin/env python3
"""
Athena Bot - Controlled Wild Trading Bot
Uses configuration from config.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.bot.unified_bot import UnifiedTradingBot

class AthenaBot(UnifiedTradingBot):
    def __init__(self):
        super().__init__("athena")

def main():
    bot = AthenaBot()
    bot.run()

if __name__ == "__main__":
    main()