#!/usr/bin/env python3
"""
Quick test of the Anti-FOMO Apollo improvements
Runs a few cycles to see how the new logic performs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Apollo.apollo import SimpleApollo

def test_anti_fomo():
    print("🧪 Testing Anti-FOMO Apollo Bot")
    print("=" * 50)
    
    # Create bot with same settings but will use new anti-FOMO logic
    bot = SimpleApollo(
        ml_threshold=0.85,
        stop_loss_pct=0.004,
        trailing_stop_pct=0.003,
        override_threshold=0.92,
        take_profit_pct=0.008,
        cooldown_minutes=5,
        max_override_loss_pct=0.005,
        min_hold_minutes=10
    )
    
    print("🔍 Running 5 test cycles to evaluate new logic...")
    
    # Run a few test cycles
    for i in range(5):
        print(f"\n--- Test Cycle {i+1} ---")
        try:
            bot.run_cycle()
        except Exception as e:
            print(f"Error in cycle {i+1}: {e}")
        
        # Small delay between cycles
        import time
        time.sleep(2)
    
    print("\n✅ Test completed!")
    print("📊 Check the logs to see how the anti-FOMO filters are working")
    print("🎯 Key improvements:")
    print("   - Won't buy at recent highs (within 0.2% of 20-bar high)")
    print("   - Won't buy when RSI > 65 (overbought)")
    print("   - Won't buy when >0.5% above SMA20 (overextended)")
    print("   - Won't buy after recent 0.5% pump in 5min")
    print("   - Technical analysis now favors pullbacks over breakouts")
    print("   - Penalties for buying at tops, bonuses for buying pullbacks")

if __name__ == "__main__":
    test_anti_fomo()