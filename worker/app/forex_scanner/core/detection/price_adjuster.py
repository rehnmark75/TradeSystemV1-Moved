# core/detection/price_adjuster.py
"""
Price Adjustment Module
Handles BID/ASK/MID price adjustments with proper pip size support
"""

import pandas as pd
import logging
from typing import Dict
try:
    import config
except ImportError:
    from forex_scanner import config


class PriceAdjuster:
    """Handles price adjustments for accurate signal detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def adjust_bid_to_mid_prices(self, df: pd.DataFrame, spread_pips: float, epic: str = None) -> pd.DataFrame:
        """
        Convert BID prices to approximate MID prices
        
        Args:
            df: DataFrame with BID prices
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            DataFrame with adjusted MID prices
        """
        # Get pip size for this pair
        pip_size = self.get_pip_size(epic) if epic else 0.0001
        spread = spread_pips * pip_size  # Convert pips to price
        
        df_adjusted = df.copy()
        df_adjusted['open'] = df['open'] + spread/2
        df_adjusted['high'] = df['high'] + spread/2
        df_adjusted['low'] = df['low'] + spread/2
        df_adjusted['close'] = df['close'] + spread/2
        
        self.logger.debug(f"Adjusted prices by {spread_pips} pips spread")
        
        return df_adjusted
    
    def get_pip_size(self, epic: str = None) -> float:
        """
        Get pip size for specific epic (compatible with forex_optimizer)
        
        Args:
            epic: Epic code
            
        Returns:
            Pip size (0.01 for JPY pairs, 0.0001 for most others)
        """
        if not epic:
            return 0.0001  # Default to standard pip size
            
        epic_upper = epic.upper()
        
        # Check if it's a JPY pair
        if 'JPY' in epic_upper:
            return 0.01  # JPY pairs use 2 decimal places
        # Check for other special cases
        elif any(metal in epic_upper for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 0.01  # Metals often use 2 decimal places
        else:
            return 0.0001  # Standard forex pairs use 4 decimal places
    
    def get_bid_price(self, mid_price: float, spread_pips: float, epic: str = None) -> float:
        """
        Calculate BID price from MID price
        
        Args:
            mid_price: Mid price
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            BID price
        """
        pip_size = self.get_pip_size(epic)
        spread = spread_pips * pip_size
        return mid_price - (spread / 2)
    
    def get_ask_price(self, mid_price: float, spread_pips: float, epic: str = None) -> float:
        """
        Calculate ASK price from MID price
        
        Args:
            mid_price: Mid price
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            ASK price
        """
        pip_size = self.get_pip_size(epic)
        spread = spread_pips * pip_size
        return mid_price + (spread / 2)
    
    def adjust_entry_price_buy(self, price: float, spread_pips: float, epic: str = None) -> float:
        """
        Calculate buy entry price (ASK price) from mid price
        
        Args:
            price: Mid price
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            Buy entry price (ASK)
        """
        return self.get_ask_price(price, spread_pips, epic)

    def adjust_entry_price_sell(self, price: float, spread_pips: float, epic: str = None) -> float:
        """
        Calculate sell entry price (BID price) from mid price
        
        Args:
            price: Mid price  
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            Sell entry price (BID)
        """
        return self.get_bid_price(price, spread_pips, epic)

    def calculate_execution_price(self, price: float, signal_type: str, spread_pips: float, epic: str = None) -> float:
        """
        Calculate execution price based on signal type
        
        Args:
            price: Mid price
            signal_type: 'BULL'/'BUY' or 'BEAR'/'SELL'
            spread_pips: Spread in pips
            epic: Epic code for pair-specific pip calculation
            
        Returns:
            Execution price
        """
        if signal_type.upper() in ['BULL', 'BUY']:
            return self.adjust_entry_price_buy(price, spread_pips, epic)
        else:
            return self.adjust_entry_price_sell(price, spread_pips, epic)
    
    def get_pip_multiplier(self, epic: str) -> int:
        """
        Get pip multiplier for specific epic (legacy compatibility)
        
        Args:
            epic: Epic code
            
        Returns:
            Pip multiplier (10000 for most pairs, 100 for JPY pairs)
        """
        pip_size = self.get_pip_size(epic)
        return int(1 / pip_size)
    
    def calculate_pip_distance(self, price1: float, price2: float, epic: str) -> float:
        """
        Calculate pip distance between two prices
        
        Args:
            price1: First price
            price2: Second price
            epic: Epic code for pip multiplier
            
        Returns:
            Distance in pips
        """
        pip_size = self.get_pip_size(epic)
        return abs(price1 - price2) / pip_size
    
    def add_execution_prices(self, signal: Dict, spread_pips: float) -> Dict:
        """
        Add execution prices for trading signals
        
        Args:
            signal: Signal dictionary
            spread_pips: Spread in pips
            
        Returns:
            Updated signal with execution prices
        """
        epic = signal.get('epic', '')
        pip_size = self.get_pip_size(epic)
        spread = spread_pips * pip_size  # Convert to price
        current_price_mid = signal['price']
        
        if signal['signal_type'] in ['BULL', 'BUY']:
            execution_price = current_price_mid + (spread / 2)  # ASK price for buying
        else:  # BEAR/SELL
            execution_price = current_price_mid - (spread / 2)  # BID price for selling
        
        signal.update({
            'price_mid': current_price_mid,
            'execution_price': execution_price,
            'spread_pips': spread_pips
        })
        
        return signal