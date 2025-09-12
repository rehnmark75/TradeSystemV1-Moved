# core/signal_detector.py
"""
EMA Signal Detection System
Clean implementation of EMA crossover strategy with BID/MID price handling
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.behavior import BehaviorAnalyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer
import config


class SignalDetector:
    """EMA-based signal detection with comprehensive market analysis"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.data_fetcher = DataFetcher(db_manager)
        self.technical_analyzer = TechnicalAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        
        self.logger = logging.getLogger(__name__)
    
    def detect_signals_bid_adjusted(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5
    ) -> Optional[Dict]:
        """
        Detect EMA signals with BID price adjustment to MID prices
        
        Args:
            epic: Epic code
            pair: Currency pair name
            spread_pips: Spread in pips
            
        Returns:
            Signal dictionary or None if no signal detected
        """
        try:
            # Get enhanced data
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, timeframe='5m')
            
            if df_5m is None or len(df_5m) < config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Adjust BID prices to MID prices
            df_mid = self._adjust_bid_to_mid_prices(df_5m, spread_pips)
            
            # Add EMA indicators
            df_with_emas = self.technical_analyzer.add_ema_indicators(df_mid, config.EMA_PERIODS)
            
            # Detect signal
            signal = self._detect_ema_crossover_signal(df_with_emas, epic)
            
            if signal:
                # Add execution prices
                signal = self._add_execution_prices(signal, spread_pips)
                # Add enhanced market context
                signal = self._add_market_context(signal, df_with_emas)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting BID-adjusted signals for {epic}: {e}")
            return None
    
    def detect_signals_mid_prices(self, epic: str, pair: str) -> Optional[Dict]:
        """
        Detect EMA signals using MID prices (no adjustment needed)
        
        Args:
            epic: Epic code
            pair: Currency pair name
            
        Returns:
            Signal dictionary or None if no signal detected
        """
        try:
            # Get enhanced data
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, timeframe='5m')
            
            if df_5m is None or len(df_5m) < config.MIN_BARS_FOR_SIGNAL:
                return None
            
            # Add EMA indicators directly (assuming MID prices)
            df_with_emas = self.technical_analyzer.add_ema_indicators(df_5m, config.EMA_PERIODS)
            
            # Detect signal
            signal = self._detect_ema_crossover_signal(df_with_emas, epic)
            
            if signal:
                # Add enhanced market context
                signal = self._add_market_context(signal, df_with_emas)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error detecting MID-price signals for {epic}: {e}")
            return None
    
    def _adjust_bid_to_mid_prices(self, df: pd.DataFrame, spread_pips: float) -> pd.DataFrame:
        """Convert BID prices to approximate MID prices"""
        spread = spread_pips / config.PAIR_INFO.get('default', {}).get('pip_multiplier', 10000)
        
        df_adjusted = df.copy()
        df_adjusted['open'] = df['open'] + spread/2
        df_adjusted['high'] = df['high'] + spread/2
        df_adjusted['low'] = df['low'] + spread/2
        df_adjusted['close'] = df['close'] + spread/2
        
        return df_adjusted
    
    def _detect_ema_crossover_signal(self, df: pd.DataFrame, epic: str) -> Optional[Dict]:
        """
        Core EMA crossover signal detection logic
        
        Args:
            df: DataFrame with EMA indicators
            epic: Epic code
            
        Returns:
            Signal dictionary or None
        """
        if len(df) < 2:
            return None
        
        # Get latest and previous values
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        current_price = latest['close']
        prev_price = previous['close']
        
        ema_9_current = latest['ema_9']
        ema_21_current = latest['ema_21']
        ema_200_current = latest['ema_200']
        ema_9_prev = previous['ema_9']
        
        # Check for BULL signal
        bull_conditions = {
            'price_above_ema9': current_price > ema_9_current,
            'ema9_above_ema21': ema_9_current > ema_21_current,
            'ema9_above_ema200': ema_9_current > ema_200_current,
            'ema21_above_ema200': ema_21_current > ema_200_current,
            'new_crossover': prev_price <= ema_9_prev and current_price > ema_9_current
        }
        
        # Check for BEAR signal
        bear_conditions = {
            'price_below_ema9': current_price < ema_9_current,
            'ema21_above_ema9': ema_21_current > ema_9_current,
            'ema200_above_ema9': ema_200_current > ema_9_current,
            'ema200_above_ema21': ema_200_current > ema_21_current,
            'new_crossover': prev_price >= ema_9_prev and current_price < ema_9_current
        }
        
        # Determine signal type and calculate confidence
        if all(bull_conditions.values()):
            signal_type = 'BULL'
            confidence = self._calculate_confidence(latest, 'BULL')
        elif all(bear_conditions.values()):
            signal_type = 'BEAR'
            confidence = self._calculate_confidence(latest, 'BEAR')
        else:
            return None
        
        # Create signal dictionary
        signal = {
            'signal_type': signal_type,
            'epic': epic,
            'timeframe': '5m',
            'timestamp': latest['start_time'],
            'price': current_price,
            'ema_9': ema_9_current,
            'ema_21': ema_21_current,
            'ema_200': ema_200_current,
            'confidence_score': confidence,
            'conditions_met': bull_conditions if signal_type == 'BULL' else bear_conditions
        }
        
        return signal
    
    def _calculate_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on EMA separation and volume"""
        current_price = latest_row['close']
        ema_9 = latest_row['ema_9']
        ema_21 = latest_row['ema_21']
        
        # Base confidence
        base_confidence = 0.5
        
        # EMA separation bonus (stronger separation = higher confidence)
        if signal_type == 'BULL':
            ema_separation = abs(ema_9 - ema_21) / current_price * 10000  # pips
        else:
            ema_separation = abs(ema_21 - ema_9) / current_price * 10000  # pips
        
        separation_bonus = min(0.3, ema_separation * 0.02)  # Max 30% bonus
        
        # Volume bonus (if volume data available)
        volume_bonus = 0
        if 'volume_ratio_20' in latest_row:
            volume_ratio = latest_row.get('volume_ratio_20', 1.0)
            volume_bonus = min(0.15, max(0, (volume_ratio - 1.0) * 0.15))  # Max 15% bonus
        
        # Combined confidence (max 95%)
        confidence = min(0.95, base_confidence + separation_bonus + volume_bonus)
        
        return confidence
    
    def _add_execution_prices(self, signal: Dict, spread_pips: float) -> Dict:
        """Add execution prices for BID-adjusted signals"""
        spread = spread_pips / 10000  # Convert to decimal
        current_price_mid = signal['price']
        
        if signal['signal_type'] == 'BULL':
            execution_price = current_price_mid + (spread / 2)  # ASK price for buying
        else:  # BEAR
            execution_price = current_price_mid - (spread / 2)  # BID price for selling
        
        signal.update({
            'price_mid': current_price_mid,
            'execution_price': execution_price,
            'spread_pips': spread_pips
        })
        
        return signal
    
    def _add_market_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Add enhanced market context to signal"""
        latest = df.iloc[-1]
        
        # Add available market context
        context_fields = [
            'volume_ratio_20', 'distance_to_support_pips', 'distance_to_resistance_pips',
            'trend_alignment', 'consolidation_range_pips', 'bars_since_breakout',
            'rejection_wicks_count', 'consecutive_green_candles', 'consecutive_red_candles'
        ]
        
        for field in context_fields:
            if field in latest:
                signal[field] = latest[field]
        
        # Add recent candle data for context
        signal['enhanced_data'] = df.tail(5).to_dict('records')
        
        return signal
    
    def backtest_signals(
        self,
        epic_list: List[str],
        lookback_days: int = 30,
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5
    ) -> List[Dict]:
        """
        Backtest EMA signals on historical data
        
        Args:
            epic_list: List of epics to test
            lookback_days: Days of historical data to analyze
            use_bid_adjustment: Whether to use BID price adjustment
            spread_pips: Spread for BID adjustment
            
        Returns:
            List of historical signals found
        """
        self.logger.info(f"🔙 Starting backtest: {len(epic_list)} pairs, {lookback_days} days")
        
        all_signals = []
        
        for epic in epic_list:
            try:
                pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
                pair = pair_info['pair']
                
                self.logger.info(f"Backtesting {epic} ({pair})...")
                
                # Get historical data
                df_5m = self.data_fetcher.get_enhanced_data(
                    epic, pair, timeframe='5m', 
                    lookback_hours=lookback_days * 24
                )
                
                if df_5m is None or len(df_5m) < config.MIN_BARS_FOR_SIGNAL:
                    self.logger.warning(f"Insufficient data for {epic}")
                    continue
                
                # Get historical signals
                if use_bid_adjustment:
                    signals = self._backtest_epic_bid_adjusted(df_5m, epic, spread_pips)
                else:
                    signals = self._backtest_epic_mid_prices(df_5m, epic)
                
                all_signals.extend(signals)
                self.logger.info(f"Found {len(signals)} signals for {epic}")
                
            except Exception as e:
                self.logger.error(f"Error backtesting {epic}: {e}")
                continue
        
        # Sort by timestamp
        all_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        self.logger.info(f"📊 Backtest complete: {len(all_signals)} total signals")
        return all_signals
    
    def _backtest_epic_bid_adjusted(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float
    ) -> List[Dict]:
        """Backtest single epic with BID adjustment"""
        # Adjust to MID prices
        df_mid = self._adjust_bid_to_mid_prices(df, spread_pips)
        
        # Add EMAs
        df_with_emas = self.technical_analyzer.add_ema_indicators(df_mid, config.EMA_PERIODS)
        
        return self._scan_historical_signals(df_with_emas, epic, use_bid_adjustment=True, spread_pips=spread_pips)
    
    def _backtest_epic_mid_prices(self, df: pd.DataFrame, epic: str) -> List[Dict]:
        """Backtest single epic with MID prices"""
        # Add EMAs directly
        df_with_emas = self.technical_analyzer.add_ema_indicators(df, config.EMA_PERIODS)
        
        return self._scan_historical_signals(df_with_emas, epic, use_bid_adjustment=False)
    
    def _scan_historical_signals(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        use_bid_adjustment: bool = False,
        spread_pips: float = 1.5
    ) -> List[Dict]:
        """Scan DataFrame for historical signals"""
        signals = []
        start_idx = config.MIN_BARS_FOR_SIGNAL  # Start after we have enough data for 200 EMA
        
        for i in range(start_idx + 1, len(df)):
            try:
                # Create mini DataFrame for signal detection
                current_data = df.iloc[i-1:i+1].copy()  # Previous and current rows
                
                if len(current_data) < 2:
                    continue
                
                # Detect signal
                signal = self._detect_ema_crossover_signal(current_data, epic)
                
                if signal:
                    # Add execution prices if using BID adjustment
                    if use_bid_adjustment:
                        signal = self._add_execution_prices(signal, spread_pips)
                    
                    # Add performance metrics
                    signal = self._add_backtest_performance(signal, df, i)
                    
                    signals.append(signal)
                    
            except Exception as e:
                continue  # Skip problematic rows
        
        return signals
    
    def _add_backtest_performance(self, signal: Dict, df: pd.DataFrame, signal_index: int) -> Dict:
        """Add performance metrics for backtesting"""
        try:
            # Look ahead 20 bars to calculate potential profit/loss
            future_bars = df.iloc[signal_index:signal_index + 20]
            
            if len(future_bars) == 0:
                return signal
            
            current_price = signal['price']
            signal_type = signal['signal_type']
            
            if signal_type == 'BULL':
                max_profit = (future_bars['high'].max() - current_price) * 10000  # pips
                max_loss = (current_price - future_bars['low'].min()) * 10000  # pips
            else:  # BEAR
                max_profit = (current_price - future_bars['low'].min()) * 10000  # pips
                max_loss = (future_bars['high'].max() - current_price) * 10000  # pips
            
            signal.update({
                'max_profit_pips': max_profit,
                'max_loss_pips': max_loss,
                'risk_reward_ratio': max_profit / max_loss if max_loss > 0 else 0
            })
            
        except Exception as e:
            pass  # Don't fail the signal if performance calculation fails
        
        return signal
    
    def analyze_performance(self, signals: List[Dict]) -> Dict:
        """Analyze performance of historical signals"""
        if not signals:
            return {}
        
        total_signals = len(signals)
        bull_signals = [s for s in signals if s['signal_type'] == 'BULL']
        bear_signals = [s for s in signals if s['signal_type'] == 'BEAR']
        
        # Calculate metrics
        avg_confidence = sum(s.get('confidence_score', 0) for s in signals) / total_signals
        
        # Performance metrics (if available)
        signals_with_performance = [s for s in signals if 'max_profit_pips' in s]
        
        if signals_with_performance:
            avg_profit = sum(s['max_profit_pips'] for s in signals_with_performance) / len(signals_with_performance)
            avg_loss = sum(s['max_loss_pips'] for s in signals_with_performance) / len(signals_with_performance)
            
            # Win rate (assuming 20 pip target, 10 pip stop)
            profit_target = 20
            stop_loss = 10
            
            winners = [s for s in signals_with_performance if s['max_profit_pips'] >= profit_target]
            losers = [s for s in signals_with_performance if s['max_loss_pips'] >= stop_loss]
            
            win_rate = len(winners) / len(signals_with_performance) if signals_with_performance else 0
        else:
            avg_profit = avg_loss = win_rate = 0
        
        performance = {
            'total_signals': total_signals,
            'bull_signals': len(bull_signals),
            'bear_signals': len(bear_signals),
            'average_confidence': avg_confidence,
            'average_profit_pips': avg_profit,
            'average_loss_pips': avg_loss,
            'win_rate': win_rate,
            'signals_with_performance': len(signals_with_performance)
        }
        
        # Log performance summary
        self.logger.info("📈 Performance Analysis:")
        self.logger.info(f"  Total signals: {total_signals}")
        self.logger.info(f"  Bull/Bear: {len(bull_signals)}/{len(bear_signals)}")
        self.logger.info(f"  Avg confidence: {avg_confidence:.1%}")
        if signals_with_performance:
            self.logger.info(f"  Avg profit: {avg_profit:.1f} pips")
            self.logger.info(f"  Avg loss: {avg_loss:.1f} pips")
            self.logger.info(f"  Win rate: {win_rate:.1%}")
        
        return performance