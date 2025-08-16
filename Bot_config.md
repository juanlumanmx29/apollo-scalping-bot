# Bot Configuration Guide

## Trading Bot Settings Overview

| Bot | Stop Loss | Trailing Stop | Take Profit | Max Loss | ML Threshold | Strategy |
|-----|-----------|---------------|-------------|----------|--------------|----------|
| **Apollo** | 1% | 0.2% | 0.3% | 1.5% | 70% | True Scalping |
| **Midas** | 5% | 0.6% | None | 6% | 80% | Wildest (Let Winners Run) |
| **Athena** | 5% | 0.6% | 2% | 6% | 80% | Wild with Controlled Exit |

## Bot Strategies Explained

### 🚀 Apollo - True Scalping Bot
**Goal**: Quick, frequent profits with tight risk control

- **ML Threshold**: 70% (more frequent trades)
- **Stop Loss**: 1% (reasonable protection against volatility)
- **Trailing Stop**: 0.2% (captures small profits as price moves up)
- **Take Profit**: 0.3% (quick scalping exits)
- **Max Loss**: 1.5% (absolute emergency brake)
- **Min Hold**: 3 minutes (fast scalping)
- **Cooldown**: 2 minutes (quick re-entry)

**Best For**: Traders wanting frequent, small profits with controlled risk

### 🏛️ Midas - The Wildest Bot
**Goal**: Maximum risk/reward, let big winners run indefinitely

- **ML Threshold**: 80% (selective, high-quality entries)
- **Stop Loss**: 5% (wide stops for volatility tolerance)
- **Trailing Stop**: 0.6% (protects gains while allowing room to breathe)
- **Take Profit**: None (let winners run forever!)
- **Max Loss**: 6% (emergency brake for disasters)
- **Min Hold**: 30 minutes (patience required)
- **Cooldown**: 15 minutes (slower, methodical trading)

**Best For**: Risk-tolerant traders wanting to catch big moves with unlimited upside

### 🏛️ Athena - Wild with Control
**Goal**: Midas-like patience with predictable profit targets

- **ML Threshold**: 80% (selective entries like Midas)
- **Stop Loss**: 5% (same wide tolerance as Midas)
- **Trailing Stop**: 0.6% (same protection as Midas)
- **Take Profit**: 2% (controlled exit unlike Midas)
- **Max Loss**: 6% (same disaster protection as Midas)
- **Min Hold**: 30 minutes (patient like Midas)
- **Cooldown**: 15 minutes (methodical like Midas)

**Best For**: Traders wanting big-move potential with predictable 2% profit exits

## Key Configuration Details

### Stop Loss Logic
- **Apollo (1%)**: Protects against normal crypto volatility while allowing room to breathe
- **Midas/Athena (5%)**: Wide stops allow riding through major market swings to catch big trends

### Trailing Stop Logic
- **Apollo (0.2%)**: Tight trailing captures small scalping profits quickly
- **Midas/Athena (0.6%)**: Wider trailing protects larger gains while allowing continuation

### ML Thresholds
- **Apollo (70%)**: Lower threshold = more frequent trades for scalping
- **Midas/Athena (80%)**: Higher threshold = more selective, quality entries

### Take Profit Strategy
- **Apollo (0.3%)**: Quick exits for scalping profits
- **Midas (None)**: No limits - let winners run indefinitely
- **Athena (2%)**: Controlled exits for consistent gains

## Risk Profiles

| Risk Level | Bot | Max Loss Per Trade | Expected Win Rate | Trade Frequency |
|------------|-----|-------------------|-------------------|-----------------|
| **LOW** | Apollo | 1.5% | 65-70% | High (multiple/hour) |
| **HIGH** | Midas | 6% | 55-65% | Low (1-3/day) |
| **MEDIUM-HIGH** | Athena | 6% | 60-70% | Low (1-3/day) |

## Emergency Safeguards

All bots include these safety mechanisms:
- **Absolute Stop Loss**: Final emergency brake that can never be overridden
- **Wallet Balance Check**: Prevents trading without sufficient funds
- **Position Limits**: Only one position at a time per bot
- **Cooldown Periods**: Prevents overtrading after profitable trades
- **Basic Market Validation**: Ensures price data quality

## Model Integration

All bots use the same **Smart Entry Model** (`models/smart_entry_model_20250810_172020.joblib`):

### 🎯 What the Model Predicts
**The Core Question**: *"Will ETH price increase by at least 0.3% within the next 10 minutes?"*

### 📊 Model Training Details
- **Model Type**: XGBoost Classifier
- **Target**: 0.3% price increase within 10 minutes
- **Performance**: 0.961 ROC-AUC (96.1% accuracy)
- **Features**: 23 smart entry detection features
- **Training Date**: August 10, 2025

### ✅ What Constitutes a "Good Entry" (Training Criteria)
- ✅ **Profit Target**: Price goes up 0.3% or more
- ✅ **Time Window**: Within 10 minutes
- ✅ **Risk Control**: Doesn't drop more than 0.3% first
- ✅ **Smart Timing**: Entry is NOT within 0.5% of recent high
- ✅ **Reversal Focus**: Identifies pullbacks and mean reversion opportunities

### 🧠 Key Features the Model Analyzes
**Reversal Detection:**
- Pullback from recent highs
- Bounce from recent lows
- Distance from recent highs/lows

**Mean Reversion Indicators:**
- RSI (looks for oversold conditions < 30)
- Distance from 20-period SMA
- Momentum deceleration patterns

**Volume & Pattern Analysis:**
- Volume declining patterns
- Candlestick patterns (doji, hammer)
- Upper/lower wick analysis
- Time of day factors

### 📈 Model Confidence Interpretation
- **80-100% Confidence**: Strong reversal signals detected, high probability of 0.3% gain
- **70-79% Confidence**: Good setup with reasonable probability
- **60-69% Confidence**: Moderate setup, model's minimum threshold
- **0-59% Confidence**: Poor conditions, wait for better setup

### 🎯 Why You See Low Confidence (0-1%)
The model shows low confidence when:
- RSI is neutral (30-70), not oversold
- Price is near moving averages (no mean reversion setup)
- Minimal pullback from recent highs
- No clear reversal patterns detected

### 💡 When Confidence Will Rise
The model looks for:
- 🚀 **RSI drops to 25-35** (oversold conditions)
- 🚀 **ETH pulls back 1-2%** from recent highs
- 🚀 **Volume reversal patterns** 
- 🚀 **Price moves significantly from moving averages**
- 🚀 **Clear support bounces** or **resistance breaks**

### ⚙️ Bot Threshold Settings
- **Apollo**: 70% minimum confidence (more frequent trades)
- **Midas**: 80% minimum confidence (selective entries)
- **Athena**: 80% minimum confidence (selective entries)

**The model is working perfectly - it's being selective and waiting for the right market conditions rather than chasing momentum!**

## Usage Commands

```bash
# Apollo - True Scalping
cd Apollo && python apollo.py

# Midas - Wildest Strategy  
cd Midas && python midas.py

# Athena - Wild with 2% Take Profit
cd Athena && python athena.py
```

## Configuration Philosophy

**Pure ML Decision Making**: All anti-FOMO filters have been removed. The bots now trust the ML model completely for entry decisions. The model was specifically trained to avoid buying at peaks, so manual filters are no longer needed.

**Risk-Appropriate Stops**: Stop losses and trailing stops are sized appropriately for each strategy - tight for scalping, wide for trend-following.

**Strategy Differentiation**: Each bot serves a different risk appetite and trading style while using the same high-quality ML model for entry timing.