#!/usr/bin/env python3
"""
Advanced Scalping Model Training System v2
Trains multiple model types on 1-minute data for actual scalping performance
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Feature engineering
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

from src.data_collector.binance_collector import BinanceDataCollector

class AdvancedScalpingTrainer:
    def __init__(self):
        self.collector = BinanceDataCollector()
        self.models = {}
        self.results = {}
        
    def collect_training_data(self, days=30):
        """Collect extensive 1-minute data for training"""
        print(f"📊 Collecting {days} days of 1-minute data...")
        
        all_data = []
        chunk_size = 1000  # Binance limit
        
        # Collect in chunks
        for i in range(0, days * 24 * 60, chunk_size):
            print(f"  Collecting chunk {i//chunk_size + 1}...", end='\r')
            df = self.collector.get_recent_data('1m', minutes=min(chunk_size, days * 24 * 60 - i))
            if df is not None and len(df) > 0:
                all_data.append(df)
            else:
                break
        
        if all_data:
            df = pd.concat(all_data).sort_index()
            df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
            print(f"\n✅ Collected {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df
        else:
            print("❌ Failed to collect data")
            return None
    
    def engineer_scalping_features(self, df):
        """Engineer features specifically for scalping (1-5 minute movements)"""
        print("🔧 Engineering scalping-specific features...")
        
        # Basic price features
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_3m'] = df['close'].pct_change(3)
        df['price_change_5m'] = df['close'].pct_change(5)
        
        # NEW: Mean reversion features
        df['sma_20'] = df['close'].rolling(20).mean()
        df['distance_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['sma_50'] = df['close'].rolling(50).mean()
        df['distance_from_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # NEW: Position in range (to avoid buying at tops)
        df['high_20'] = df['close'].rolling(20).max()
        df['low_20'] = df['close'].rolling(20).min()
        df['position_in_range'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)
        
        # NEW: Pullback indicators
        df['pullback_from_high_5'] = (df['high'].rolling(5).max() - df['close']) / df['close']
        df['pullback_from_high_10'] = (df['high'].rolling(10).max() - df['close']) / df['close']
        
        # Micro-structure features (very important for scalping)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Volume features (critical for detecting real moves)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # Buy/Sell pressure
        df['buy_pressure'] = df['volume'] * df['close_position']
        df['sell_pressure'] = df['volume'] * (1 - df['close_position'])
        df['pressure_ratio'] = df['buy_pressure'] / (df['sell_pressure'] + 1e-10)
        
        # Momentum features (short-term)
        df['momentum_1m'] = df['close'] - df['close'].shift(1)
        df['momentum_3m'] = df['close'] - df['close'].shift(3)
        df['momentum_5m'] = df['close'] - df['close'].shift(5)
        df['momentum_acceleration'] = df['momentum_1m'] - df['momentum_1m'].shift(1)
        
        # Technical indicators (fast versions for scalping)
        # RSI - very short period
        df['rsi_6'] = RSIIndicator(df['close'], window=6).rsi()
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
        
        # MACD - faster settings
        macd = MACD(df['close'], window_slow=12, window_fast=6, window_sign=4)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands - tighter
        bb = BollingerBands(df['close'], window=10, window_dev=1.5)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # ATR for volatility
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=7)
        df['atr'] = atr.average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Recent high/low
        df['dist_from_high_5m'] = (df['high'].rolling(5).max() - df['close']) / df['close']
        df['dist_from_low_5m'] = (df['close'] - df['low'].rolling(5).min()) / df['close']
        
        # Price patterns (micro)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['bullish_pattern'] = df['higher_high'] & df['higher_low']
        
        # Order flow imbalance (approximation)
        df['delta'] = df['close'] - df['open']
        df['cumulative_delta'] = df['delta'].rolling(10).sum()
        
        # Time features (some periods are better for scalping)
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        # Market regime
        df['volatility_regime'] = df['atr_ratio'].rolling(60).mean()
        df['trend_strength'] = abs(df['price_change_5m'].rolling(20).mean())
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"✅ Created {len(df.columns)} features")
        return df
    
    def create_scalping_target(self, df, target_pct=0.003, periods=5):
        """Create improved target that considers both profit potential and risk"""
        print(f"🎯 Creating risk-adjusted target: {target_pct:.1%} move in {periods} minutes...")
        
        # Calculate future prices
        future_prices = []
        for i in range(1, periods + 1):
            future_prices.append(df['close'].shift(-i))
        
        # Get max and min prices in next 'periods' minutes
        future_max = pd.concat(future_prices, axis=1).max(axis=1)
        future_min = pd.concat(future_prices, axis=1).min(axis=1)
        
        # Calculate potential gain and loss
        potential_gain = (future_max - df['close']) / df['close']
        potential_loss = (df['close'] - future_min) / df['close']
        
        # NEW: Risk-adjusted target
        # Good trade = reaches target AND doesn't drop too much first
        good_trade = (potential_gain >= target_pct) & (potential_loss < 0.002)  # Max 0.2% drawdown
        
        # NEW: Also penalize buying at highs
        position_in_range = df['position_in_range'] if 'position_in_range' in df.columns else 0.5
        not_at_high = position_in_range < 0.8  # Not in top 20% of range
        
        # Final target: good trade AND not buying at high
        df['target'] = (good_trade & not_at_high).astype(int)
        
        # Also create a quality score for weighting samples
        df['sample_weight'] = 1.0
        df.loc[position_in_range > 0.9, 'sample_weight'] = 0.5  # Penalize extreme highs
        df.loc[position_in_range < 0.2, 'sample_weight'] = 1.5  # Reward buying at lows
        
        # Remove last 'periods' rows (no future data)
        df = df[:-periods]
        
        # Calculate target statistics
        target_pct_actual = df['target'].mean() * 100
        print(f"✅ Target created: {target_pct_actual:.1f}% positive samples")
        print(f"   Risk-adjusted: Max drawdown < 0.2%, not at highs")
        
        return df
    
    def prepare_features(self, df):
        """Select and prepare final features for training"""
        # Features to use for scalping
        feature_cols = [
            # Price action
            'price_change_1m', 'price_change_3m', 'price_change_5m',
            'high_low_ratio', 'close_position', 'upper_shadow', 'lower_shadow',
            
            # NEW: Mean reversion features
            'distance_from_sma20', 'distance_from_sma50',
            'position_in_range', 'pullback_from_high_5', 'pullback_from_high_10',
            
            # Volume
            'volume_ratio', 'volume_trend', 'volume_spike',
            'buy_pressure', 'sell_pressure', 'pressure_ratio',
            
            # Momentum
            'momentum_1m', 'momentum_3m', 'momentum_5m', 'momentum_acceleration',
            
            # Technical
            'rsi_6', 'rsi_14', 'macd', 'macd_signal', 'macd_diff',
            'bb_position', 'atr_ratio',
            
            # Market structure
            'dist_from_high_5m', 'dist_from_low_5m',
            'higher_high', 'higher_low', 'bullish_pattern',
            'delta', 'cumulative_delta',
            
            # Time
            'hour', 'is_london_session', 'is_ny_session', 'is_asian_session',
            
            # Regime
            'volatility_regime', 'trend_strength'
        ]
        
        # Ensure all features exist
        available_features = [f for f in feature_cols if f in df.columns]
        
        X = df[available_features].copy()
        y = df['target'].copy()
        
        print(f"✅ Using {len(available_features)} features for training")
        return X, y, available_features
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        self.evaluate_model(best_rf, X_test, y_test, "Random Forest")
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        self.evaluate_model(best_xgb, X_test, y_test, "XGBoost")
        
        return best_xgb
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Deep Neural Network"""
        print("\n🧠 Training Neural Network...")
        
        # Scale features for NN
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with class weights for imbalanced data
        class_weight = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            class_weight=class_weight,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print(f"Neural Network - Test Accuracy: {np.mean(y_pred == y_test):.3f}")
        print(f"Neural Network - ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Store model with scaler
        self.models['neural_network'] = {'model': model, 'scaler': scaler}
        self.results['neural_network'] = roc_auc_score(y_test, y_pred_proba)
        
        return model, scaler
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble of multiple models"""
        print("\n🎭 Training Ensemble Model...")
        
        # Train base models with lower complexity to avoid overfitting
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        xgb = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        lr = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
        
        # Fit base models
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        # Get predictions from each model
        rf_pred = rf.predict_proba(X_test)[:, 1]
        xgb_pred = xgb.predict_proba(X_test)[:, 1]
        lr_pred = lr.predict_proba(X_test)[:, 1]
        
        # Weighted average ensemble
        ensemble_pred = (0.4 * rf_pred + 0.4 * xgb_pred + 0.2 * lr_pred)
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        print(f"Ensemble - Test Accuracy: {np.mean(ensemble_binary == y_test):.3f}")
        print(f"Ensemble - ROC AUC: {roc_auc_score(y_test, ensemble_pred):.3f}")
        
        self.models['ensemble'] = {'rf': rf, 'xgb': xgb, 'lr': lr, 'weights': [0.4, 0.4, 0.2]}
        self.results['ensemble'] = roc_auc_score(y_test, ensemble_pred)
        
        return self.models['ensemble']
    
    def evaluate_model(self, model, X_test, y_test, name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {np.mean(y_pred == y_test):.3f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            print(f"\nTop 10 Features:")
            print(importances)
        
        self.models[name.lower().replace(' ', '_')] = model
        self.results[name.lower().replace(' ', '_')] = roc_auc_score(y_test, y_pred_proba)
    
    def save_best_model(self, X, feature_names):
        """Save the best performing model"""
        if not self.results:
            print("❌ No models trained")
            return
        
        # Find best model
        best_model_name = max(self.results, key=self.results.get)
        best_score = self.results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name} (ROC AUC: {best_score:.3f})")
        
        # Prepare model data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if best_model_name == 'neural_network':
            # Save neural network separately
            model_data = self.models[best_model_name]
            model_data['model'].save(f'models/scalping_nn_model_{timestamp}.h5')
            
            # Save scaler and metadata
            model_info = {
                'model_type': 'neural_network',
                'scaler': model_data['scaler'],
                'feature_names': feature_names,
                'threshold': 0.5,
                'target_percentage': 0.3,
                'prediction_periods': 5,
                'roc_auc': best_score,
                'timestamp': timestamp
            }
            joblib.dump(model_info, f'models/scalping_nn_metadata_{timestamp}.joblib')
            
        else:
            # Save sklearn models
            model_info = {
                'model': self.models[best_model_name],
                'model_type': best_model_name,
                'feature_names': feature_names,
                'threshold': 0.5,
                'target_percentage': 0.3,
                'prediction_periods': 5,
                'roc_auc': best_score,
                'timestamp': timestamp
            }
            
            if best_model_name == 'ensemble':
                # Special handling for ensemble
                model_info['model'] = self.models[best_model_name]
            
            # Save with multiple names
            joblib.dump(model_info, f'models/scalping_model_{best_model_name}_{timestamp}.joblib')
            joblib.dump(model_info, f'models/scalping_model_latest.joblib')
            joblib.dump(model_info, f'models/scalping_model_5min.joblib')  # Override old model
            
        print(f"✅ Model saved successfully!")
        print(f"   - Type: {best_model_name}")
        print(f"   - Features: {len(feature_names)}")
        print(f"   - Performance: {best_score:.3f} ROC AUC")
    
    def backtest_final_model(self, model, X, y, feature_names):
        """Perform final backtesting with trading simulation"""
        print("\n📊 Backtesting best model...")
        
        # Prepare for backtesting
        if isinstance(model, dict) and 'model' in model:
            # Neural network
            nn_model = model['model']
            scaler = model['scaler']
            X_scaled = scaler.transform(X)
            probabilities = nn_model.predict(X_scaled).flatten()
        elif isinstance(model, dict) and 'rf' in model:
            # Ensemble
            rf_pred = model['rf'].predict_proba(X)[:, 1]
            xgb_pred = model['xgb'].predict_proba(X)[:, 1]
            lr_pred = model['lr'].predict_proba(X)[:, 1]
            weights = model['weights']
            probabilities = weights[0] * rf_pred + weights[1] * xgb_pred + weights[2] * lr_pred
        else:
            # Single model
            probabilities = model.predict_proba(X)[:, 1]
        
        # Simulate trading with different thresholds
        thresholds = [0.6, 0.7, 0.8, 0.9]
        
        print("\nBacktest Results by Threshold:")
        print("Threshold | Trades | Win Rate | Avg Confidence")
        print("-" * 50)
        
        for threshold in thresholds:
            signals = probabilities >= threshold
            n_trades = signals.sum()
            
            if n_trades > 0:
                win_rate = y[signals].mean() * 100
                avg_conf = probabilities[signals].mean()
                print(f"{threshold:.1f}      | {n_trades:6d} | {win_rate:6.1f}%  | {avg_conf:.3f}")
            else:
                print(f"{threshold:.1f}      | {n_trades:6d} | N/A      | N/A")
        
        # Recommend optimal threshold
        valid_thresholds = []
        for threshold in thresholds:
            signals = probabilities >= threshold
            if signals.sum() >= 20:  # At least 20 trades
                win_rate = y[signals].mean()
                if win_rate > 0.6:  # At least 60% win rate
                    valid_thresholds.append((threshold, win_rate, signals.sum()))
        
        if valid_thresholds:
            # Choose threshold with best balance of win rate and number of trades
            best_threshold = max(valid_thresholds, key=lambda x: x[1] * np.log(x[2]))
            print(f"\n✅ Recommended threshold: {best_threshold[0]:.1f} ({best_threshold[1]:.1%} win rate, {best_threshold[2]} trades)")
        else:
            print("\n⚠️  No threshold found with >60% win rate and sufficient trades")

def main():
    """Main training pipeline"""
    print("🚀 Advanced Scalping Model Training System v2")
    print("=" * 60)
    
    trainer = AdvancedScalpingTrainer()
    
    # Step 1: Collect data
    df = trainer.collect_training_data(days=14)  # 2 weeks of data
    if df is None:
        return
    
    # Step 2: Engineer features
    df = trainer.engineer_scalping_features(df)
    
    # Step 3: Create target (0.3% move in 5 minutes)
    df = trainer.create_scalping_target(df, target_pct=0.003, periods=5)
    
    # Step 4: Prepare features
    X, y, feature_names = trainer.prepare_features(df)
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print(f"📊 Target distribution: {y_train.mean():.1%} positive")
    
    # Step 6: Train models
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.train_neural_network(X_train, y_train, X_test, y_test)
    trainer.train_ensemble(X_train, y_train, X_test, y_test)
    
    # Step 7: Save best model
    trainer.save_best_model(X, feature_names)
    
    # Step 8: Backtest
    best_model_name = max(trainer.results, key=trainer.results.get)
    trainer.backtest_final_model(trainer.models[best_model_name], X, y, feature_names)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()