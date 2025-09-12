#!/usr/bin/env python3
"""
Fast EMA Strategy for Optimization
Simplified EMA strategy with reduced validation layers for ultra-fast parameter optimization

Key optimizations:
- 2-layer validation (vs 5-layer full)
- No Two-Pole Oscillator analysis
- No Momentum Bias Index
- No Multi-timeframe validation
- Minimal data requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.strategies.base_strategy import BaseStrategy

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class FastEMAStrategy(BaseStrategy):
    """
    Ultra-fast EMA strategy for parameter optimization
    
    Validation layers (2 total):
    1. EMA crossover detection (core signal)
    2. EMA 200 trend filter (basic trend alignment)
    
    Removed for speed:
    - Two-Pole Oscillator (15m + 1H)
    - Momentum Bias Index
    - MACD validation
    - Multi-timeframe analysis
    - Complex confidence calculations
    """
    
    def __init__(self, ema_config: dict = None, min_confidence: float = 0.45):
        self.name = 'fast_ema'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Simple EMA configuration
        self.ema_config = ema_config or {'short': 21, 'long': 50, 'trend': 200}
        self.ema_short = self.ema_config.get('short', 21)
        self.ema_long = self.ema_config.get('long', 50) 
        self.ema_trend = self.ema_config.get('trend', 200)
        
        # Basic parameters
        self.min_confidence = min_confidence
        self.min_bars = max(self.ema_trend + 10, 50)  # Minimum bars for EMA stability
        
        self.logger.info(f"🚀 Fast EMA Strategy initialized - Periods: {self.ema_short}/{self.ema_long}/{self.ema_trend}")
        self.logger.info(f"   Validation: SIMPLIFIED (2 layers for speed)")
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only essential EMA indicators for speed"""
        df_enhanced = df.copy()
        
        # Add only the EMAs we need (no extra indicators)
        df_enhanced[f'ema_{self.ema_short}'] = df_enhanced['close'].ewm(span=self.ema_short).mean()
        df_enhanced[f'ema_{self.ema_long}'] = df_enhanced['close'].ewm(span=self.ema_long).mean()
        df_enhanced[f'ema_{self.ema_trend}'] = df_enhanced['close'].ewm(span=self.ema_trend).mean()
        
        # Semantic mapping for compatibility
        df_enhanced['ema_short'] = df_enhanced[f'ema_{self.ema_short}']
        df_enhanced['ema_long'] = df_enhanced[f'ema_{self.ema_long}']
        df_enhanced['ema_trend'] = df_enhanced[f'ema_{self.ema_trend}']
        
        return df_enhanced
    
    def detect_signal_auto(self, 
                          df: pd.DataFrame,
                          epic: str = None,
                          spread_pips: float = 0,
                          timeframe: str = '15m',
                          **kwargs) -> Optional[Dict]:
        """
        Fast signal detection with minimal validation
        
        Args:
            df: Price data with indicators
            epic: Trading pair 
            spread_pips: Bid/ask spread
            timeframe: Timeframe
            
        Returns:
            Signal dict or None
        """
        if df is None or df.empty or len(df) < self.min_bars:
            return None
        
        # Add indicators
        df_with_indicators = self.add_indicators(df)
        
        # Get latest values
        latest = df_with_indicators.iloc[-1]
        prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else latest
        
        try:
            # LAYER 1: EMA Crossover Detection
            crossover_signal = self._detect_ema_crossover(df_with_indicators)
            if not crossover_signal:
                return None
            
            signal_type = crossover_signal['type']  # 'BUY' or 'SELL'
            entry_price = latest['close']
            
            # LAYER 2: EMA 200 Trend Filter
            if not self._validate_trend_alignment(latest, signal_type):
                return None
            
            # Simple confidence calculation (no complex features)
            confidence = self._calculate_simple_confidence(df_with_indicators, signal_type)
            
            if confidence < self.min_confidence:
                return None
            
            # Create signal
            signal = {
                'signal_type': signal_type,
                'confidence': confidence,
                'price': entry_price,
                'timestamp': latest.name if hasattr(latest, 'name') else 'now',
                'epic': epic,
                'timeframe': timeframe,
                'strategy': 'fast_ema',
                'ema_config': f"{self.ema_short}/{self.ema_long}/{self.ema_trend}",
                
                # Technical details
                'ema_short_value': latest['ema_short'],
                'ema_long_value': latest['ema_long'],
                'ema_trend_value': latest['ema_trend'],
                'crossover_strength': abs(latest['ema_short'] - latest['ema_long']) / entry_price,
                
                # Fast mode indicators
                'validation_layers': 2,
                'fast_mode': True
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Fast signal detection failed: {e}")
            return None
    
    def _detect_ema_crossover(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect EMA crossover with minimal complexity"""
        if len(df) < 2:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Current EMA relationship
        current_short_above = latest['ema_short'] > latest['ema_long']
        prev_short_above = prev['ema_short'] > prev['ema_long']
        
        # Detect crossover
        if current_short_above and not prev_short_above:
            # Bullish crossover: short EMA crossed above long EMA
            return {'type': 'BUY', 'crossover': 'bullish'}
        elif not current_short_above and prev_short_above:
            # Bearish crossover: short EMA crossed below long EMA  
            return {'type': 'SELL', 'crossover': 'bearish'}
        
        return None
    
    def _validate_trend_alignment(self, latest_row, signal_type: str) -> bool:
        """Basic trend validation using EMA 200"""
        price = latest_row['close']
        ema_200 = latest_row['ema_trend']
        
        if signal_type == 'BUY':
            # For buy signals, prefer price above EMA 200 (uptrend)
            return price > ema_200
        else:  # SELL
            # For sell signals, prefer price below EMA 200 (downtrend)
            return price < ema_200
    
    def _calculate_simple_confidence(self, df: pd.DataFrame, signal_type: str) -> float:
        """Simple confidence calculation for speed"""
        if len(df) < 10:
            return 0.0
        
        latest = df.iloc[-1]
        
        try:
            # Base confidence
            base_confidence = 0.50
            
            # EMA alignment bonus (all EMAs in order)
            if signal_type == 'BUY':
                ema_aligned = (latest['ema_short'] > latest['ema_long'] > latest['ema_trend'])
            else:
                ema_aligned = (latest['ema_short'] < latest['ema_long'] < latest['ema_trend'])
            
            if ema_aligned:
                base_confidence += 0.15
            
            # Price momentum bonus (simple)
            recent_bars = df.tail(5)
            price_momentum = (recent_bars['close'].iloc[-1] - recent_bars['close'].iloc[0]) / recent_bars['close'].iloc[0]
            
            if signal_type == 'BUY' and price_momentum > 0:
                base_confidence += min(0.20, price_momentum * 100)  # Cap at 20% bonus
            elif signal_type == 'SELL' and price_momentum < 0:
                base_confidence += min(0.20, abs(price_momentum) * 100)
            
            # Crossover strength bonus
            crossover_strength = abs(latest['ema_short'] - latest['ema_long']) / latest['close']
            base_confidence += min(0.10, crossover_strength * 1000)  # Cap at 10% bonus
            
            return min(0.95, base_confidence)  # Cap at 95%
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {e}")
            return 0.50  # Safe default
    
    def detect_signal(self, df: pd.DataFrame, epic: str = None, timeframe: str = '15m', **kwargs):
        """
        Required method from BaseStrategy interface
        Delegates to detect_signal_auto for backwards compatibility
        """
        return self.detect_signal_auto(df, epic, timeframe=timeframe, **kwargs)
    
    def get_required_indicators(self) -> list:
        """Required indicators for fast EMA strategy"""
        return [
            f'ema_{self.ema_short}',
            f'ema_{self.ema_long}', 
            f'ema_{self.ema_trend}'
        ]


def create_fast_ema_strategy(ema_config: dict = None, min_confidence: float = 0.45) -> FastEMAStrategy:
    """Factory function to create fast EMA strategy"""
    return FastEMAStrategy(ema_config=ema_config, min_confidence=min_confidence)


# Test function
if __name__ == "__main__":
    import time
    
    # Test the fast strategy
    print("🧪 TESTING FAST EMA STRATEGY")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=300, freq='15T')
    prices = 100 + np.cumsum(np.random.randn(300) * 0.1)
    
    test_df = pd.DataFrame({
        'start_time': dates,
        'open': prices + np.random.randn(300) * 0.05,
        'high': prices + abs(np.random.randn(300) * 0.1),
        'low': prices - abs(np.random.randn(300) * 0.1),  
        'close': prices,
        'volume': np.random.randint(100, 1000, 300)
    }).set_index('start_time')
    
    # Test fast strategy performance
    strategy = FastEMAStrategy()
    
    start_time = time.time()
    signal = strategy.detect_signal_auto(test_df, epic='TEST.PAIR')
    end_time = time.time()
    
    print(f"✅ Fast strategy test completed")
    print(f"   Processing time: {(end_time - start_time)*1000:.2f}ms")
    print(f"   Signal detected: {signal is not None}")
    if signal:
        print(f"   Signal type: {signal['signal_type']}")
        print(f"   Confidence: {signal['confidence']:.0%}")
        print(f"   Validation layers: {signal['validation_layers']}")
        print(f"   Fast mode: {signal['fast_mode']}")
    
    print(f"\n🚀 Fast EMA Strategy ready for optimization!")