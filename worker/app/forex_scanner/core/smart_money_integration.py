# core/smart_money_integration.py
"""
MINIMAL Smart Money Integration for Existing SignalDetector - FIXED VERSION
This file adds smart money analysis WITHOUT modifying your existing signal_detector.py

FIXES:
- Proper handling of None signals
- Safe signal copying
- Better error messages
- Graceful degradation when signals are invalid

Integration approach:
1. Your existing SignalDetector remains 100% unchanged
2. This module adds a simple wrapper method that can be called AFTER signal detection
3. No architectural changes needed - just add smart money context to existing signals
4. Can be enabled/disabled with a single config flag
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import threading

from .smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
from .database import DatabaseManager
# Import utilities for JSON serialization
try:
    from utils.scanner_utils import make_json_serializable, clean_signal_for_json
    import config
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable, clean_signal_for_json
    from forex_scanner import config


class SmartMoneyIntegration:
    """
    Minimal smart money integration that works alongside your existing SignalDetector
    FIXED: Proper null handling for signals
    """
    
    def __init__(self, database_manager: DatabaseManager, data_fetcher):
        self.logger = logging.getLogger(__name__)
        self.db_manager = database_manager
        self.data_fetcher = data_fetcher
        
        # Initialize smart money analyzer
        try:
            self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(data_fetcher)
        except Exception as e:
            self.logger.warning(f"Failed to initialize SmartMoneyReadOnlyAnalyzer: {e}")
            self.smart_money_analyzer = None
        
        # Configuration
        self.enabled = getattr(config, 'SMART_MONEY_READONLY_ENABLED', True)
        self.store_enhanced_signals = getattr(config, 'STORE_ENHANCED_SIGNALS', True)
        
        self.logger.info("🧠 SmartMoneyIntegration initialized")
        self.logger.info(f"   Status: {'✅ ENABLED' if self.enabled else '❌ DISABLED'}")
        self.logger.info(f"   Analyzer available: {'✅' if self.smart_money_analyzer else '❌'}")
    
    def enhance_signal_with_smart_money(
        self,
        signal: Optional[Dict],
        epic: str,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Enhance a single signal with smart money analysis
        FIXED: Proper handling of None signals
        
        Args:
            signal: Original signal from your existing detector (can be None)
            epic: Trading pair epic
            timeframe: Analysis timeframe
            
        Returns:
            Enhanced signal with smart money data, or original signal if enhancement fails
        """
        # FIXED: Check if signal is None first
        if signal is None:
            self.logger.debug(f"Signal is None for {epic}, skipping smart money enhancement")
            return None
        
        if not self.enabled:
            return signal
        
        if not self.smart_money_analyzer:
            self.logger.debug("Smart money analyzer not available")
            return signal
        
        try:
            # FIXED: Create a safe copy of the signal
            enhanced_signal = signal.copy() if isinstance(signal, dict) else signal
            
            # Get data for this epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # Check if we have a data fetcher
            if not self.data_fetcher:
                self.logger.debug("Data fetcher not available for smart money analysis")
                return enhanced_signal
            
            # Get price data
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe)
            
            if df is None or len(df) < 100:
                self.logger.debug(f"Insufficient data for smart money analysis: {epic}")
                return enhanced_signal
            
            # Run smart money analysis
            smart_money_result = self.smart_money_analyzer.analyze_signal(
                enhanced_signal, df, epic, timeframe
            )
            
            if smart_money_result and isinstance(smart_money_result, dict):
                # Merge smart money results into signal
                enhanced_signal['smart_money_validated'] = smart_money_result.get('smart_money_validated', False)
                enhanced_signal['smart_money_type'] = smart_money_result.get('smart_money_type', 'UNKNOWN')
                enhanced_signal['smart_money_score'] = smart_money_result.get('smart_money_score', 0.5)
                enhanced_signal['enhanced_confidence_score'] = smart_money_result.get(
                    'enhanced_confidence_score', 
                    enhanced_signal.get('confidence_score', 0.5)
                )
                
                # Add analysis details as JSON strings for database storage
                # FIXED: Add analysis details as JSON strings for database storage using make_json_serializable
                if smart_money_result.get('market_structure_analysis'):
                    market_structure = smart_money_result['market_structure_analysis']
                    enhanced_signal['market_structure_analysis'] = json.dumps(
                        make_json_serializable(market_structure)
                    )
                
                if smart_money_result.get('order_flow_analysis'):
                    order_flow = smart_money_result['order_flow_analysis']
                    enhanced_signal['order_flow_analysis'] = json.dumps(
                        make_json_serializable(order_flow)
                    )
                
                if smart_money_result.get('confluence_details'):
                    confluence = smart_money_result['confluence_details']
                    enhanced_signal['confluence_details'] = json.dumps(
                        make_json_serializable(confluence)
                    )
                
                # FIXED: Handle metadata with proper serialization
                if smart_money_result.get('analysis_metadata'):
                    metadata = smart_money_result['analysis_metadata']
                    enhanced_signal['smart_money_metadata'] = json.dumps(
                        make_json_serializable(metadata)
                    )

                
                self.logger.debug(f"✅ Smart money enhancement successful for {epic}")
            else:
                self.logger.debug(f"Smart money analysis returned no results for {epic}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Smart money enhancement failed for {epic}: {e}")
            # Return original signal on error
            return signal
    
    def enhance_signals_batch(
        self,
        signals: Optional[List[Dict]],
        default_timeframe: str = '5m'
    ) -> List[Dict]:
        """
        Enhance multiple signals with smart money analysis
        FIXED: Proper handling of None and invalid signal lists
        
        Args:
            signals: List of signals to enhance (can be None or empty)
            default_timeframe: Default timeframe for analysis
            
        Returns:
            List of enhanced signals
        """
        # FIXED: Handle None or invalid signals
        if signals is None:
            return []
        
        if not isinstance(signals, list):
            self.logger.warning(f"Signals is not a list: {type(signals)}")
            return []
        
        if not signals:
            return []
        
        if not self.enabled or not self.smart_money_analyzer:
            return signals
        
        enhanced_signals = []
        
        for signal in signals:
            # Skip None signals in the list
            if signal is None:
                continue
            
            # Skip non-dictionary signals
            if not isinstance(signal, dict):
                self.logger.warning(f"Skipping non-dictionary signal: {type(signal)}")
                enhanced_signals.append(signal)
                continue
            
            epic = signal.get('epic', '')
            timeframe = signal.get('timeframe', default_timeframe)
            
            if epic:
                enhanced = self.enhance_signal_with_smart_money(signal, epic, timeframe)
                if enhanced:
                    enhanced_signals.append(enhanced)
            else:
                # No epic, can't enhance
                enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def store_enhanced_signal(
        self,
        signal: Optional[Dict],
        alert_id: Optional[int] = None
    ) -> bool:
        """
        Store enhanced signal with smart money data
        FIXED: Proper null checking
        
        Args:
            signal: Enhanced signal to store
            alert_id: Optional alert ID if already saved
            
        Returns:
            Success status
        """
        if not self.store_enhanced_signals:
            return False
        
        # FIXED: Check for None signal
        if signal is None:
            self.logger.debug("Cannot store None signal")
            return False
        
        if not isinstance(signal, dict):
            self.logger.warning(f"Cannot store non-dictionary signal: {type(signal)}")
            return False
        
        if not self.db_manager:
            return False
        
        try:
            # Implementation would update the database with smart money fields
            # This is a placeholder - actual implementation depends on your database schema
            self.logger.debug(f"Storing enhanced signal for {signal.get('epic', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced signal: {e}")
            return False


def add_smart_money_to_signal(
    signal: Optional[Dict], 
    epic: str, 
    data_fetcher, 
    db_manager
) -> Optional[Dict]:
    """
    Standalone function to add smart money analysis to a single signal
    Can be called from anywhere in your existing code
    FIXED: Proper null handling
    
    Args:
        signal: Original signal (can be None)
        epic: Trading pair
        data_fetcher: Data fetcher instance
        db_manager: Database manager instance
        
    Returns:
        Enhanced signal or original signal if enhancement fails
    """
    # FIXED: Return None if signal is None
    if signal is None:
        return None
    
    try:
        if not getattr(config, 'SMART_MONEY_READONLY_ENABLED', True):
            return signal
        
        integration = SmartMoneyIntegration(db_manager, data_fetcher)
        return integration.enhance_signal_with_smart_money(signal, epic)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Smart money integration failed: {e}")
        return signal


def add_smart_money_to_signals(
    signals: Optional[List[Dict]], 
    data_fetcher, 
    db_manager
) -> List[Dict]:
    """
    Standalone function to add smart money analysis to multiple signals
    Can be called from anywhere in your existing code
    FIXED: Proper null handling
    
    Args:
        signals: List of signals (can be None or empty)
        data_fetcher: Data fetcher instance
        db_manager: Database manager instance
        
    Returns:
        List of enhanced signals
    """
    # FIXED: Return empty list if signals is None
    if signals is None:
        return []
    
    try:
        if not getattr(config, 'SMART_MONEY_READONLY_ENABLED', True):
            return signals
        
        integration = SmartMoneyIntegration(db_manager, data_fetcher)
        return integration.enhance_signals_batch(signals)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Smart money batch integration failed: {e}")
        return signals


# MINIMAL INTEGRATION FOR YOUR EXISTING METHODS
def enhance_your_existing_signal_methods():
    """
    Instructions for minimal integration with your existing SignalDetector methods
    
    To integrate smart money analysis, you only need to add 2-3 lines to your existing methods:
    
    BEFORE (your existing code):
    ```python
    def detect_signals_bid_adjusted(self, epic: str, pair: str, spread_pips: float = 1.5, timeframe: str = '5m'):
        # ... your existing logic ...
        signal = self.ema_strategy.detect_signal(df, epic, spread_pips, timeframe)
        if signal:
            signal = self._add_market_context(signal, df)
        return signal
    ```
    
    AFTER (with smart money integration):
    ```python
    def detect_signals_bid_adjusted(self, epic: str, pair: str, spread_pips: float = 1.5, timeframe: str = '5m'):
        # ... your existing logic ... (UNCHANGED)
        signal = self.ema_strategy.detect_signal(df, epic, spread_pips, timeframe)
        if signal:
            signal = self._add_market_context(signal, df)
            # ADD THESE 2 LINES (with null check):
            from .smart_money_integration import add_smart_money_to_signal
            signal = add_smart_money_to_signal(signal, epic, self.data_fetcher, self.db_manager)
        return signal
    ```
    
    That's it! Your existing methods remain 99% unchanged.
    """
    pass


# Example usage in signal_detector.py methods
def example_integration_in_detect_signals_all_strategies(self, epic, pair, spread_pips, timeframe):
    """
    Example of how to integrate in detect_signals_all_strategies method
    """
    # Your existing code that collects signals...
    all_signals = []
    
    # ... existing signal detection logic ...
    
    # After collecting all signals, enhance them with smart money
    if all_signals:
        # FIXED: Safe integration that handles None values
        from .smart_money_integration import add_smart_money_to_signals
        all_signals = add_smart_money_to_signals(all_signals, self.data_fetcher, self.db_manager)
    
    return all_signals


# Configuration additions for config.py
"""
Add these minimal configurations to your config.py file:

# Smart Money Read-Only Integration (Minimal)
SMART_MONEY_READONLY_ENABLED = True          # Enable/disable smart money analysis
STORE_ENHANCED_SIGNALS = True                # Store enhanced signals in database
SMART_MONEY_PROCESSING_TIMEOUT = 5.0         # Analysis timeout in seconds
SMART_MONEY_MIN_DATA_POINTS = 100           # Minimum data points required
SMART_MONEY_STRUCTURE_WEIGHT = 0.4          # Weight for market structure (0-1)
SMART_MONEY_ORDER_FLOW_WEIGHT = 0.6         # Weight for order flow (0-1)
"""