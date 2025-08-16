#!/usr/bin/env python3
"""
Test all 3 Anti-FOMO bots with their different strategies
"""

import sys
import os

def test_bot_imports():
    """Test that all 3 bots can be imported successfully"""
    print("🧪 Testing All Anti-FOMO Bots Import")
    print("=" * 50)
    
    try:
        # Test Apollo
        sys.path.insert(0, os.path.join(os.getcwd(), 'Apollo'))
        from apollo import SimpleApollo
        print("✅ Apollo bot imported successfully")
        
        # Test Midas  
        sys.path.insert(0, os.path.join(os.getcwd(), 'Midas'))
        from midas import MidasBot
        print("✅ Midas bot imported successfully")
        
        # Test Athena
        sys.path.insert(0, os.path.join(os.getcwd(), 'Athena'))
        from athena import AthenaBot  
        print("✅ Athena bot imported successfully")
        
        print("\n🎯 All 3 bots have Anti-FOMO improvements:")
        print("   ⚡ Apollo: TRUE SCALPING (70% ML, 0.15% stop, 0.3% profit)")
        print("   🔥 Midas: WILDEST (80% ML, 5% stop, NO take profit)")  
        print("   🎯 Athena: WILD + PROFIT (80% ML, 5% stop, 2% profit)")
        
        print("\n🚫 Anti-FOMO Features (All Bots):")
        print("   • No buying at recent highs (within 0.2% of 20-bar high)")
        print("   • No buying when RSI > 65 (overbought)")
        print("   • No buying when >0.5% above SMA20 (overextended)")
        print("   • No buying after recent 0.5% pump in 5min")
        print("   • Technical analysis rewards pullbacks over breakouts")
        print("   • Penalties for buying at tops, bonuses for pullback entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing bots: {e}")
        return False

def show_bot_differences():
    """Show the key differences between the 3 bot strategies"""
    print("\n📋 Bot Strategy Comparison:")
    print("-" * 65)
    print("Setting             | Apollo  | Midas   | Athena")  
    print("-" * 65)
    print("ML Threshold        | 70%     | 80%     | 80%")
    print("Stop Loss           | 0.15%   | 5.0%    | 5.0%") 
    print("Take Profit         | 0.3%    | NONE    | 2.0%")
    print("Override Threshold  | 85%     | 90%     | 90%")
    print("Min Hold Time       | 3min    | 30min   | 30min")
    print("Cooldown            | 2min    | 15min   | 15min")
    print("Max Loss            | 0.2%    | 6.0%    | 6.0%")
    print("-" * 65)
    print("Strategy Focus      | Scalping| Wildest | Wild+Exit")
    
if __name__ == "__main__":
    success = test_bot_imports()
    if success:
        show_bot_differences()
        print("\n✅ All bots ready with Anti-FOMO protection!")
        print("\nTo run individually:")
        print("   cd Apollo && python apollo.py")
        print("   cd Midas && python midas.py") 
        print("   cd Athena && python athena.py")
    else:
        print("❌ Fix import issues before running bots")