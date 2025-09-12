# ================================
# 5. analysis/signals.py
# ================================
import pandas as pd
from typing import List, Dict, Optional
from typing import Optional
from core.data_structures import Signal, SignalType
from core.config import EpicConfig
from analysis.technical import TechnicalAnalysis

# ================================
# 5. SIGNAL DETECTION ENGINE
# ================================

class SignalDetector:
    """Advanced signal detection with multiple strategies"""
    
    def __init__(self, technical_analysis: TechnicalAnalysis):
        self.ta = technical_analysis
    
    def detect_ema_signals(self, df: pd.DataFrame, epic: str) -> Optional[Signal]:
        """Detect EMA-based signals"""
        if len(df) < 200:
            return None
        
        # Get epic settings
        settings = EpicConfig.get_settings(epic)
        spread_pips = settings['spread_pips']
        pip_multiplier = settings['pip_multiplier']
        
        # Adjust for spread (convert BID to MID if needed)
        df_adjusted = self._adjust_for_spread(df, spread_pips, pip_multiplier)
        
        # Add EMAs
        df_with_emas = self.ta.add_ema_indicators(df_adjusted)
        
        latest = df_with_emas.iloc[-1]
        previous = df_with_emas.iloc[-2]
        
        # Check for valid EMA values
        if any(pd.isna([latest['ema_9'], latest['ema_21'], latest['ema_200']])):
            return None
        
        # Bull signal conditions
        bull_conditions = {
            'price_above_ema9': latest['close'] > latest['ema_9'],
            'ema9_above_ema21': latest['ema_9'] > latest['ema_21'],
            'ema9_above_ema200': latest['ema_9'] > latest['ema_200'],
            'ema21_above_ema200': latest['ema_21'] > latest['ema_200'],
            'new_signal': previous['close'] <= previous['ema_9'] and latest['close'] > latest['ema_9']
        }
        
        # Bear signal conditions
        bear_conditions = {
            'price_below_ema9': latest['close'] < latest['ema_9'],
            'ema21_above_ema9': latest['ema_21'] > latest['ema_9'],
            'ema200_above_ema9': latest['ema_200'] > latest['ema_9'],
            'ema200_above_ema21': latest['ema_200'] > latest['ema_21'],
            'new_signal': previous['close'] >= previous['ema_9'] and latest['close'] < latest['ema_9']
        }
        
        # Determine signal type and confidence
        signal_type = None
        confidence_score = 0
        
        if all(bull_conditions.values()):
            signal_type = SignalType.BULL
            confidence_score = self._calculate_confidence(latest, 'bull', settings)
        elif all(bear_conditions.values()):
            signal_type = SignalType.BEAR
            confidence_score = self._calculate_confidence(latest, 'bear', settings)
        
        if signal_type:
            return Signal(
                signal_type=signal_type,
                epic=epic,
                timestamp=latest['start_time'],
                price=latest['close'],
                confidence_score=confidence_score,
                ema_9=latest['ema_9'],
                ema_21=latest['ema_21'],
                ema_200=latest['ema_200'],
                spread_pips=spread_pips,
                pip_multiplier=pip_multiplier,
                volume_ratio=latest.get('volume_ratio_20'),
                distance_to_support_pips=latest.get('distance_to_support_pips'),
                distance_to_resistance_pips=latest.get('distance_to_resistance_pips'),
                trend_alignment=latest.get('trend_short')
            )
        
        return None
    
    def _adjust_for_spread(self, df: pd.DataFrame, spread_pips: float, pip_multiplier: int) -> pd.DataFrame:
        """Adjust BID prices to approximate MID prices"""
        spread = spread_pips / pip_multiplier
        df_adjusted = df.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            df_adjusted[col] = df[col] + spread / 2
        
        return df_adjusted
    
    def _calculate_confidence(self, latest_row: pd.Series, signal_direction: str, settings: Dict) -> float:
        """Calculate signal confidence score"""
        base_confidence = 0.5
        
        # EMA separation bonus
        if signal_direction == 'bull':
            ema_separation = (latest_row['ema_9'] - latest_row['ema_21']) / latest_row['close']
        else:
            ema_separation = (latest_row['ema_21'] - latest_row['ema_9']) / latest_row['close']
        
        separation_bonus = min(0.2, ema_separation * 1000)  # Scale appropriately
        
        # Volume bonus
        volume_ratio = latest_row.get('volume_ratio_20', 1.0)
        volume_bonus = min(0.15, max(0, (volume_ratio - 1) * 0.15))
        
        # Volatility adjustment
        volatility_adj = {
            'low': 0.1, 'medium': 0.05, 'high': 0.0, 'very_high': -0.05
        }.get(settings['volatility'], 0)
        
        confidence = base_confidence + separation_bonus + volume_bonus + volatility_adj
        return min(0.95, max(0.1, confidence))

