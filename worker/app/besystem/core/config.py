# ================================
# 2. EPIC CONFIGURATION
# ================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


class EpicConfig:
    """Epic-specific trading configurations"""
    
    EPIC_SETTINGS = {
        'EURUSD': {'spread_pips': 1.2, 'pip_multiplier': 10000, 'min_move_pips': 0.5, 'volatility': 'medium'},
        'GBPUSD': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDJPY': {'spread_pips': 1.0, 'pip_multiplier': 100, 'min_move_pips': 1.0, 'volatility': 'medium'},
        'USDCHF': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'low'},
        'AUDUSD': {'spread_pips': 1.8, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'high'},
        'USDCAD': {'spread_pips': 2.0, 'pip_multiplier': 10000, 'min_move_pips': 0.8, 'volatility': 'medium'},
        'NZDUSD': {'spread_pips': 2.5, 'pip_multiplier': 10000, 'min_move_pips': 1.0, 'volatility': 'high'},
        'EURGBP': {'spread_pips': 1.5, 'pip_multiplier': 10000, 'min_move_pips': 0.6, 'volatility': 'low'},
        'EURJPY': {'spread_pips': 1.5, 'pip_multiplier': 100, 'min_move_pips': 1.5, 'volatility': 'medium'},
        'GBPJPY': {'spread_pips': 2.0, 'pip_multiplier': 100, 'min_move_pips': 2.0, 'volatility': 'very_high'},
    }
    
    @classmethod
    def get_settings(cls, epic: str) -> Dict:
        """Get epic-specific settings"""
        pair = cls.extract_pair_from_epic(epic)
        default_settings = {
            'spread_pips': 2.0, 
            'pip_multiplier': 10000, 
            'min_move_pips': 1.0,
            'volatility': 'medium'
        }
        return cls.EPIC_SETTINGS.get(pair, default_settings)
    
    @staticmethod
    def extract_pair_from_epic(epic: str) -> str:
        """Extract currency pair from IG epic format"""
        parts = epic.split('.')
        if len(parts) >= 3:
            return parts[2]
        return 'EURUS'