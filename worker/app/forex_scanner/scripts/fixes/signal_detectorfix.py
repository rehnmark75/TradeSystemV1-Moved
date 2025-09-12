# core/signal_detector.py - Fixed version with proper crossover detection
"""
Signal Detector Module - Fixed to prevent continuous signal generation
Ensures signals are only generated on actual crossover events
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

from .data_fetcher import DataFetcher
from .strategies.ema_strategy import EMAStrategy
from .strategies.macd_strategy import MACDStrategy
from .strategies.combined_strategy import CombinedStrategy
from .detection.price_adjuster import PriceAdjuster
from analysis.technical import TechnicalAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.behavior import BehaviorAnalyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer
import config


class SignalDetector:
    """
    Fixed SignalDetector that prevents continuous signal generation
    Tracks crossover states to ensure signals only on actual crossovers
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self.ema_strategy = EMAStrategy()
        self.macd_strategy = MACDStrategy()
        self.combined_strategy = CombinedStrategy()
        
        # Price adjustment
        self.price_adjuster = PriceAdjuster()
        
        # Technical analyzers
        self.technical = TechnicalAnalyzer()
        self.volume = VolumeAnalyzer()
        self.behavior = BehaviorAnalyzer()
        self.mtf = MultiTimeframeAnalyzer(data_fetcher)
        
        # CRITICAL: Track last crossover state to prevent duplicate signals
        self.last_crossover_state = {}  # {epic: {'ema_position': 'above/below', 'last_signal_time': datetime}}
        
        self.logger.info("✅ SignalDetector initialized with crossover tracking")
    
    def detect_signals_bid_adjusted(
        self, 
        epic: str,
        pair: str,
        config_preset: str = None,
        enable_debug: bool = False
    ) -> Optional[Dict]:
        """
        Main signal detection method with crossover tracking
        """
        try:
            # Get latest data
            timeframe = config.DEFAULT_TIMEFRAME
            df = self.data_fetcher.get_latest_data(epic, timeframe)
            
            if df.empty or len(df) < config.MIN_BARS_FOR_SIGNAL:
                if enable_debug:
                    self.logger.debug(f"Insufficient data for {epic}: {len(df)} bars")
                return None
            
            # Apply spread
            spread_pips = config.SPREAD_PIPS
            
            # Check for actual crossover event
            crossover_detected = self._check_for_crossover(df, epic)
            
            if not crossover_detected:
                if enable_debug:
                    self.logger.debug(f"No crossover detected for {epic}")
                return None
            
            # Run enabled strategies only if crossover detected
            signals = []
            
            if config.SIMPLE_EMA_STRATEGY:
                ema_signal = self.ema_strategy.detect_signal(df, epic, spread_pips, timeframe)
                if ema_signal:
                    ema_signal = self._enhance_signal(ema_signal, df, pair)
                    signals.append(ema_signal)
            
            if config.MACD_EMA_STRATEGY:
                macd_signal = self.macd_strategy.detect_signal(df, epic, spread_pips, timeframe)
                if macd_signal:
                    macd_signal = self._enhance_signal(macd_signal, df, pair)
                    signals.append(macd_signal)
            
            # Combined strategy
            if len(signals) > 0 and getattr(config, 'COMBINED_STRATEGY', True):
                combined_signal = self.combined_strategy.combine_signals(
                    signals, epic, spread_pips, timeframe
                )
                if combined_signal:
                    combined_signal = self._enhance_signal(combined_signal, df, pair)
                    signals.append(combined_signal)
            
            # Select best signal
            best_signal = self._select_best_signal(signals)
            
            if best_signal:
                # Update crossover tracking
                self._update_crossover_state(epic, best_signal['signal_type'], df)
                
                # Add timestamp
                best_signal['signal_timestamp'] = datetime.now()
                
                self.logger.info(
                    f"✅ Signal detected for {epic}: {best_signal['signal_type']} "
                    f"@ {best_signal.get('price', 'N/A')} [{best_signal['strategy']}] "
                    f"Confidence: {best_signal.get('confidence_score', 0):.1%}"
                )
                
            return best_signal
            
        except Exception as e:
            self.logger.error(f"Error detecting signals for {epic}: {e}")
            return None
    
    def _check_for_crossover(self, df: pd.DataFrame, epic: str) -> bool:
        """
        Check if an actual EMA crossover has occurred
        Prevents continuous signal generation on the same side
        """
        if len(df) < 2:
            return False
        
        # Get EMA values
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Determine EMA columns based on config
        ema_config = getattr(config, 'EMA_STRATEGY_CONFIG', {}).get(
            getattr(config, 'ACTIVE_EMA_CONFIG', 'default'),
            {'short': 9, 'long': 21, 'trend': 200}
        )
        
        ema_short_col = f"ema_{ema_config['short']}"
        
        # Check if EMAs exist
        if ema_short_col not in df.columns:
            return False
        
        current_price = latest['close']
        prev_price = previous['close']
        ema_short = latest[ema_short_col]
        ema_short_prev = previous[ema_short_col]
        
        # Determine current position
        current_position = 'above' if current_price > ema_short else 'below'
        
        # Get last known state
        last_state = self.last_crossover_state.get(epic, {})
        last_position = last_state.get('ema_position', None)
        last_signal_time = last_state.get('last_signal_time', None)
        
        # Check for actual crossover
        crossover = False
        
        if last_position is None:
            # First time checking this epic
            crossover = True
        elif last_position != current_position:
            # Position changed - crossover detected
            crossover = True
        else:
            # Same position - check if enough time has passed (prevent rapid signals)
            if last_signal_time:
                time_since_last = datetime.now() - last_signal_time
                # Allow new signal if more than 30 minutes have passed
                if time_since_last > timedelta(minutes=30):
                    # Check for fresh crossover
                    if (prev_price <= ema_short_prev and current_price > ema_short) or \
                       (prev_price >= ema_short_prev and current_price < ema_short):
                        crossover = True
        
        return crossover
    
    def _update_crossover_state(self, epic: str, signal_type: str, df: pd.DataFrame):
        """Update crossover tracking state"""
        latest = df.iloc[-1]
        
        # Get current EMA position
        ema_config = getattr(config, 'EMA_STRATEGY_CONFIG', {}).get(
            getattr(config, 'ACTIVE_EMA_CONFIG', 'default'),
            {'short': 9, 'long': 21, 'trend': 200}
        )
        ema_short_col = f"ema_{ema_config['short']}"
        
        if ema_short_col in df.columns:
            current_price = latest['close']
            ema_short = latest[ema_short_col]
            position = 'above' if current_price > ema_short else 'below'
        else:
            position = 'above' if signal_type == 'BULL' else 'below'
        
        self.last_crossover_state[epic] = {
            'ema_position': position,
            'last_signal_time': datetime.now(),
            'last_signal_type': signal_type
        }
    
    def _enhance_signal(self, signal: Dict, df: pd.DataFrame, pair: str) -> Dict:
        """Enhance signal with additional analysis"""
        try:
            # Add pair info
            signal['pair'] = pair
            
            # Add volume analysis
            volume_data = self.volume.analyze_volume_pattern(df)
            signal.update({
                'volume': volume_data.get('current_volume', 0),
                'volume_ratio': volume_data.get('volume_ratio', 1.0),
                'volume_confirmation': volume_data.get('volume_confirmation', False)
            })
            
            # Add support/resistance
            sr_levels = self.technical.find_support_resistance(df)
            latest_price = df.iloc[-1]['close']
            
            signal.update({
                'nearest_support': sr_levels.get('nearest_support', 0),
                'nearest_resistance': sr_levels.get('nearest_resistance', 0),
                'distance_to_support_pips': sr_levels.get('distance_to_support_pips', 0),
                'distance_to_resistance_pips': sr_levels.get('distance_to_resistance_pips', 0)
            })
            
            # Add market behavior
            behavior = self.behavior.analyze_market_behavior(df)
            signal.update({
                'market_session': behavior.get('session', 'Unknown'),
                'is_market_hours': behavior.get('is_market_hours', True),
                'market_regime': behavior.get('regime', 'Normal')
            })
            
            # Add multi-timeframe trend
            mtf_trend = self.mtf.analyze_trend_alignment(
                signal['epic'], 
                signal.get('timeframe', config.DEFAULT_TIMEFRAME)
            )
            signal['mtf_trend_alignment'] = mtf_trend.get('alignment_score', 0)
            
            # Calculate risk/reward
            if signal['signal_type'] == 'BULL':
                risk = abs(latest_price - signal.get('nearest_support', latest_price - 20))
                reward = abs(signal.get('nearest_resistance', latest_price + 40) - latest_price)
            else:
                risk = abs(signal.get('nearest_resistance', latest_price + 20) - latest_price)
                reward = abs(latest_price - signal.get('nearest_support', latest_price - 40))
            
            signal['risk_reward_ratio'] = reward / risk if risk > 0 else 0
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal: {e}")
            return signal
    
    def _select_best_signal(self, signals: List[Dict]) -> Optional[Dict]:
        """Select the best signal from multiple strategies"""
        if not signals:
            return None
        
        # Filter by minimum confidence
        valid_signals = [s for s in signals if s.get('confidence_score', 0) >= config.MIN_CONFIDENCE]
        
        if not valid_signals:
            return None
        
        # Sort by confidence and return best
        return max(valid_signals, key=lambda x: x.get('confidence_score', 0))
    
    def get_crossover_states(self) -> Dict:
        """Get current crossover tracking states"""
        return self.last_crossover_state.copy()
    
    def reset_crossover_state(self, epic: str = None):
        """Reset crossover state for an epic or all epics"""
        if epic:
            self.last_crossover_state.pop(epic, None)
            self.logger.info(f"Reset crossover state for {epic}")
        else:
            self.last_crossover_state.clear()
            self.logger.info("Reset all crossover states")
    
    # Additional debug methods
    def debug_signal_detection(self, epic: str, pair: str) -> Dict:
        """Debug signal detection for a specific pair"""
        debug_info = {
            'epic': epic,
            'pair': pair,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get data
            df = self.data_fetcher.get_latest_data(epic, config.DEFAULT_TIMEFRAME)
            debug_info['data_rows'] = len(df)
            
            if df.empty:
                debug_info['error'] = 'No data available'
                return debug_info
            
            # Check crossover state
            last_state = self.last_crossover_state.get(epic, {})
            debug_info['last_crossover_state'] = {
                'position': last_state.get('ema_position', 'None'),
                'last_signal_time': last_state.get('last_signal_time', 'Never').isoformat() if isinstance(last_state.get('last_signal_time'), datetime) else 'Never',
                'last_signal_type': last_state.get('last_signal_type', 'None')
            }
            
            # Check current position
            latest = df.iloc[-1]
            ema_config = getattr(config, 'EMA_STRATEGY_CONFIG', {}).get(
                getattr(config, 'ACTIVE_EMA_CONFIG', 'default'),
                {'short': 9, 'long': 21, 'trend': 200}
            )
            
            ema_short_col = f"ema_{ema_config['short']}"
            if ema_short_col in df.columns:
                current_price = latest['close']
                ema_short = latest[ema_short_col]
                current_position = 'above' if current_price > ema_short else 'below'
                
                debug_info['current_state'] = {
                    'price': float(current_price),
                    'ema_short': float(ema_short),
                    'position': current_position,
                    'crossover_detected': self._check_for_crossover(df, epic)
                }
            
            # Try detection
            signal = self.detect_signals_bid_adjusted(epic, pair, enable_debug=True)
            debug_info['signal_detected'] = signal is not None
            if signal:
                debug_info['signal'] = {
                    'type': signal.get('signal_type'),
                    'confidence': signal.get('confidence_score'),
                    'strategy': signal.get('strategy')
                }
            
        except Exception as e:
            debug_info['error'] = str(e)
        
        return debug_info