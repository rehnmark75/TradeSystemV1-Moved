# core/detection/market_conditions.py
"""
Market Conditions Analyzer
Analyzes current market conditions to guide strategy weighting
"""

import pandas as pd
from typing import Dict, Optional
import logging


class MarketConditionsAnalyzer:
    """Analyzes market conditions for dynamic strategy weighting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_market_conditions(
        self, 
        ema_signal: Optional[Dict], 
        macd_signal: Optional[Dict]
    ) -> Dict:
        """Analyze current market conditions to guide strategy weighting"""
        
        conditions = {
            'trending': False,
            'volatile': False,
            'consolidating': False,
            'volume_high': False
        }
        
        # Use any available signal for market data
        reference_signal = ema_signal or macd_signal
        if not reference_signal:
            return conditions
        
        # Check for trending conditions
        if 'ema_9' in reference_signal and 'ema_21' in reference_signal and 'ema_200' in reference_signal:
            ema_9 = reference_signal['ema_9']
            ema_21 = reference_signal['ema_21']
            ema_200 = reference_signal['ema_200']
            current_price = reference_signal['price']
            
            # Calculate EMA separation as % of price
            ema_separation_pct = abs(ema_9 - ema_21) / current_price * 100
            price_ema200_separation_pct = abs(current_price - ema_200) / current_price * 100
            
            # Trending if EMAs are well separated
            if ema_separation_pct > 0.05 and price_ema200_separation_pct > 0.1:  # 5 and 10 pip equivalent
                conditions['trending'] = True
        
        # Check for MACD trending conditions
        if 'macd_data' in reference_signal:
            macd_data = reference_signal['macd_data']
            macd_histogram = macd_data.get('macd_histogram', 0)
            
            # Strong MACD histogram indicates trending
            if abs(macd_histogram) > 0.0005:  # Threshold for strong momentum
                conditions['trending'] = True
        
        # Check volume conditions
        if 'volume_ratio_20' in reference_signal:
            volume_ratio = reference_signal['volume_ratio_20']
            if volume_ratio > 1.5:
                conditions['volume_high'] = True
        
        # Check volatility (if BB data available)
        if 'bb_width' in reference_signal:
            bb_width = reference_signal['bb_width']
            # This would need historical comparison for volatility assessment
            # For now, use a simple threshold
            if bb_width > 0.002:  # 20 pips equivalent
                conditions['volatile'] = True
        
        # Check consolidation
        if 'consolidation_range_pips' in reference_signal:
            consol_range = reference_signal['consolidation_range_pips']
            if consol_range < 10:  # Tight range
                conditions['consolidating'] = True
        
        self.logger.debug(f"Market conditions: {conditions}")
        return conditions
    
    def get_dynamic_weights(self, conditions: Dict) -> Dict[str, float]:
        """
        Get dynamic strategy weights based on market conditions
        
        Args:
            conditions: Market conditions dictionary
            
        Returns:
            Dictionary with EMA and MACD weights
        """
        base_ema_weight = 0.6
        base_macd_weight = 0.4
        
        # Adjust weights based on conditions
        if conditions['trending']:
            # In trending markets, trust EMA more
            ema_weight = min(0.8, base_ema_weight + 0.2)
            macd_weight = 1.0 - ema_weight
        elif conditions['volatile']:
            # In volatile markets, trust MACD more (momentum)
            macd_weight = min(0.7, base_macd_weight + 0.3)
            ema_weight = 1.0 - macd_weight
        elif conditions['consolidating']:
            # In consolidating markets, reduce both weights
            ema_weight = base_ema_weight * 0.8
            macd_weight = base_macd_weight * 0.8
        else:
            # Neutral market - use base weights
            ema_weight = base_ema_weight
            macd_weight = base_macd_weight
        
        return {
            'ema_weight': ema_weight,
            'macd_weight': macd_weight
        }