# analysis/multi_timeframe.py
"""
Multi-Timeframe Analysis Module
Analyzes trends across multiple timeframes for confluence
Complete implementation with all missing methods - FIXED VERSION
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from forex_scanner import config


class MultiTimeframeAnalyzer:
    """Multi-timeframe trend analysis and confluence"""
    
    def __init__(self, data_fetcher=None):
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
    
    def add_multi_timeframe_analysis(
        self, 
        df_5m: pd.DataFrame, 
        df_15m: pd.DataFrame, 
        df_1h: pd.DataFrame, 
        pair: str = 'EURUSD'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Add multi-timeframe trend analysis to dataframes
        
        Args:
            df_5m, df_15m, df_1h: DataFrames for different timeframes
            pair: Currency pair
            
        Returns:
            Tuple of enhanced dataframes
        """
        # Validate inputs
        if any(df is None or len(df) == 0 for df in [df_5m, df_15m, df_1h]):
            self.logger.warning("One or more dataframes are empty")
            return df_5m, df_15m, df_1h
        
        # Determine trends for each timeframe
        trends = self._analyze_trends_across_timeframes(df_5m, df_15m, df_1h)
        
        # Add trend data to each dataframe
        df_5m_enhanced = self._add_trend_data_to_df(df_5m, trends, '5m')
        df_15m_enhanced = self._add_trend_data_to_df(df_15m, trends, '15m')
        df_1h_enhanced = self._add_trend_data_to_df(df_1h, trends, '1h')
        
        return df_5m_enhanced, df_15m_enhanced, df_1h_enhanced
    
    def _analyze_trends_across_timeframes(
        self, 
        df_5m: pd.DataFrame, 
        df_15m: pd.DataFrame, 
        df_1h: pd.DataFrame
    ) -> Dict[str, str]:
        """Analyze trend direction across all timeframes"""
        
        trends = {
            'trend_1m': self._determine_trend(df_5m, period=4),   # Approximate 1m from 5m
            'trend_5m': self._determine_trend(df_5m, period=20),
            'trend_15m': self._determine_trend(df_15m, period=20),
            'trend_1h': self._determine_trend(df_1h, period=20),
            'trend_4h': self._determine_trend(df_1h, period=80),  # Approximate 4h from 1h
            'trend_daily': self._determine_trend(df_1h, period=240)  # Approximate daily from 1h
        }
        
        return trends
    
    def _determine_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        """
        Determine trend direction using EMA slope analysis
        
        Args:
            df: DataFrame with OHLC data
            period: EMA period for trend analysis
            
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        if df is None or len(df) < period + 10:
            return 'neutral'
        
        try:
            # Calculate EMA for trend determination
            close_col = 'close' if 'close' in df.columns else 'Close'
            if close_col not in df.columns:
                self.logger.warning("No close price column found")
                return 'neutral'
            
            ema = df[close_col].ewm(span=period).mean()
            
            # Analyze EMA slope over last 5 periods
            recent_ema = ema.tail(5)
            slope = (recent_ema.iloc[-1] - recent_ema.iloc[0]) / len(recent_ema)
            
            # Get price position relative to EMA
            current_price = df[close_col].iloc[-1]
            current_ema = ema.iloc[-1]
            price_vs_ema = (current_price - current_ema) / current_ema
            
            # Combine slope and price position for trend determination
            if slope > 0 and price_vs_ema > 0.001:  # 0.1% above EMA with positive slope
                return 'bullish'
            elif slope < 0 and price_vs_ema < -0.001:  # 0.1% below EMA with negative slope
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return 'neutral'
    
    def _add_trend_data_to_df(self, df: pd.DataFrame, trends: Dict[str, str], timeframe: str) -> pd.DataFrame:
        """Add trend data columns to dataframe"""
        
        df_enhanced = df.copy()
        
        # Add trend columns
        for trend_key, trend_value in trends.items():
            df_enhanced[trend_key] = trend_value
        
        # Add current timeframe indicator
        df_enhanced['current_timeframe'] = timeframe
        
        return df_enhanced
    
    def get_confluence_score(self, signal: Dict) -> float:
        """
        Calculate confluence score based on multi-timeframe analysis
        
        Args:
            signal: Signal dictionary with trend analysis data
            
        Returns:
            float: Confluence score between 0.0 and 1.0
        """
        try:
            base_score = signal.get('confidence_score', 0.5)
            signal_type = signal.get('signal_type', '').upper()
            
            # Get trend alignment
            trend_alignment = signal.get('trend_alignment', 'neutral')
            trend_strength = signal.get('trend_strength_score', 0.5)
            
            # Calculate confluence boost based on trend alignment
            if trend_alignment == 'bullish' and signal_type == 'BULL':
                alignment_boost = 0.2 * trend_strength
            elif trend_alignment == 'bearish' and signal_type == 'BEAR':
                alignment_boost = 0.2 * trend_strength
            else:
                alignment_boost = -0.1  # Penalty for misalignment
            
            # Calculate final confluence score
            confluence_score = min(1.0, max(0.0, base_score + alignment_boost))
            
            return confluence_score
            
        except Exception as e:
            self.logger.error(f"Error calculating confluence score: {e}")
            return signal.get('confidence_score', 0.5)
    
    def analyze_signal_confluence(
        self, 
        df: pd.DataFrame, 
        pair: str, 
        spread_pips: float = 1.5, 
        timeframe: str = '5m'
    ) -> Dict:
        """
        Analyze signal confluence across multiple strategies
        
        Args:
            df: DataFrame with market data
            pair: Currency pair
            spread_pips: Spread in pips
            timeframe: Primary timeframe
            
        Returns:
            Dict: Confluence analysis results
        """
        confluence_result = {
            'strategies_tested': [],
            'bull_signals': [],
            'bear_signals': [],
            'confluence_score': 0.0,
            'dominant_direction': 'NEUTRAL',
            'agreement_level': 'low',
            'timeframe_agreement': {}
        }
        
        try:
            # Import strategy modules dynamically to avoid circular imports
            from core.strategies.ema_strategy import EMAStrategy
            from core.strategies.macd_strategy import MACDStrategy
            
            # Test EMA strategy
            try:
                ema_strategy = EMAStrategy()
                ema_result = ema_strategy.detect_signal(df, pair)
                
                if ema_result:
                    confluence_result['strategies_tested'].append('EMA')
                    if ema_result['signal_type'] == 'BULL':
                        confluence_result['bull_signals'].append({
                            'strategy': 'EMA',
                            'confidence': ema_result.get('confidence_score', 0.0)
                        })
                    elif ema_result['signal_type'] == 'BEAR':
                        confluence_result['bear_signals'].append({
                            'strategy': 'EMA',
                            'confidence': ema_result.get('confidence_score', 0.0)
                        })
            except Exception as e:
                self.logger.debug(f"EMA strategy test failed: {e}")
            
            # Test MACD strategy
            try:
                macd_strategy = MACDStrategy()
                macd_result = macd_strategy.detect_signal(df, pair)
                
                if macd_result:
                    confluence_result['strategies_tested'].append('MACD')
                    if macd_result['signal_type'] == 'BULL':
                        confluence_result['bull_signals'].append({
                            'strategy': 'MACD',
                            'confidence': macd_result.get('confidence_score', 0.0)
                        })
                    elif macd_result['signal_type'] == 'BEAR':
                        confluence_result['bear_signals'].append({
                            'strategy': 'MACD',
                            'confidence': macd_result.get('confidence_score', 0.0)
                        })
            except Exception as e:
                self.logger.debug(f"MACD strategy test failed: {e}")
            
            # Calculate confluence metrics
            total_strategies = len(confluence_result['strategies_tested'])
            bull_count = len(confluence_result['bull_signals'])
            bear_count = len(confluence_result['bear_signals'])
            
            if total_strategies > 0:
                # Calculate confidence-weighted direction
                bull_confidence_sum = sum(signal['confidence'] for signal in confluence_result['bull_signals'])
                bear_confidence_sum = sum(signal['confidence'] for signal in confluence_result['bear_signals'])
                total_confidence = bull_confidence_sum + bear_confidence_sum
                
                # Determine dominant direction
                if bull_count > bear_count:
                    confluence_result['dominant_direction'] = 'BULL'
                elif bear_count > bull_count:
                    confluence_result['dominant_direction'] = 'BEAR'
                else:
                    confluence_result['dominant_direction'] = 'NEUTRAL'
                
                # Calculate confluence score based on agreement
                agreement_ratio = max(bull_count, bear_count) / total_strategies
                confluence_result['confluence_score'] = agreement_ratio
                
                # Add confidence weighting if we have confidence data
                if total_confidence > 0:
                    if bull_confidence_sum > bear_confidence_sum:
                        confluence_result['confidence_weighted_direction'] = 'BULL'
                    elif bear_confidence_sum > bull_confidence_sum:
                        confluence_result['confidence_weighted_direction'] = 'BEAR'
                    else:
                        confluence_result['confidence_weighted_direction'] = 'NEUTRAL'
                    
                    # Adjust confluence score based on confidence weights
                    confidence_factor = max(bull_confidence_sum, bear_confidence_sum) / total_confidence
                    confluence_result['confluence_score'] *= confidence_factor
                
                # Determine agreement level
                if confluence_result['confluence_score'] >= 0.7:
                    confluence_result['agreement_level'] = 'high'
                elif confluence_result['confluence_score'] >= 0.4:
                    confluence_result['agreement_level'] = 'medium'
                else:
                    confluence_result['agreement_level'] = 'low'
                
                # Multi-timeframe agreement (if data_fetcher is available)
                if self.data_fetcher:
                    confluence_result['timeframe_agreement'] = self._analyze_timeframe_agreement(
                        pair, confluence_result['dominant_direction']
                    )
            
            return confluence_result
            
        except Exception as e:
            self.logger.error(f"Error in confluence analysis: {e}")
            return confluence_result
    
    def _analyze_timeframe_agreement(self, pair: str, direction: str) -> Dict:
        """Analyze agreement across different timeframes"""
        timeframe_agreement = {
            '5m': 'unknown',
            '15m': 'unknown', 
            '1h': 'unknown'
        }
        
        try:
            epic = f"CS.D.{pair}.MINI.IP"
            
            # Get data for different timeframes
            timeframes = ['5m', '15m', '1h']
            lookback_hours = {'5m': 48, '15m': 168, '1h': 720}
            
            for tf in timeframes:
                try:
                    df = self.data_fetcher.get_enhanced_data(
                        epic, pair, tf, lookback_hours[tf]
                    )
                    
                    if df is not None and len(df) > 50:
                        trend = self._determine_trend(df, period=20)
                        
                        if (direction == 'BULL' and trend == 'bullish') or \
                           (direction == 'BEAR' and trend == 'bearish'):
                            timeframe_agreement[tf] = 'aligned'
                        elif trend == 'neutral':
                            timeframe_agreement[tf] = 'neutral'
                        else:
                            timeframe_agreement[tf] = 'conflicting'
                    
                except Exception as e:
                    self.logger.debug(f"Failed to analyze {tf} timeframe: {e}")
                    timeframe_agreement[tf] = 'error'
        
        except Exception as e:
            self.logger.error(f"Error in timeframe agreement analysis: {e}")
        
        return timeframe_agreement
    
    def detect_signals_multi_timeframe(
        self, 
        epic: str, 
        pair: str, 
        spread_pips: float = 1.5, 
        primary_timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Detect signals using multi-timeframe analysis
        
        Args:
            epic: Trading epic
            pair: Currency pair
            spread_pips: Spread in pips
            primary_timeframe: Primary timeframe for analysis
            
        Returns:
            Dict or None: Enhanced signal with multi-timeframe data
        """
        if not self.data_fetcher:
            self.logger.error("Data fetcher not available for multi-timeframe analysis")
            return None
        
        try:
            # Get data for multiple timeframes
            df_5m = self.data_fetcher.get_enhanced_data(epic, pair, '5m', lookback_hours=48)
            df_15m = self.data_fetcher.get_enhanced_data(epic, pair, '15m', lookback_hours=168)
            df_1h = self.data_fetcher.get_enhanced_data(epic, pair, '1h', lookback_hours=720)
            
            # Check if we have sufficient data
            min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)
            if any(df is None or len(df) < min_bars for df in [df_5m, df_15m, df_1h]):
                self.logger.debug("Insufficient data for multi-timeframe analysis")
                return None
            
            # Use primary timeframe data for signal detection
            primary_df = df_5m if primary_timeframe == '5m' else (df_15m if primary_timeframe == '15m' else df_1h)
            
            # Perform confluence analysis
            confluence_result = self.analyze_signal_confluence(
                primary_df, pair, spread_pips, primary_timeframe
            )
            
            # Check if we have a valid confluence signal
            min_confluence = getattr(config, 'MIN_CONFLUENCE_SCORE', 0.3)
            if confluence_result['confluence_score'] < min_confluence:
                return None
            
            # Create enhanced signal
            signal = {
                'epic': epic,
                'pair': pair,
                'signal_type': confluence_result['dominant_direction'],
                'timestamp': pd.Timestamp.now(),
                'timeframe': primary_timeframe,
                'confidence_score': confluence_result['confluence_score'],
                'strategy': 'Multi-Timeframe',
                'multi_timeframe_analysis': True,
                'confluence_data': confluence_result,
                'strategies_tested': confluence_result['strategies_tested'],
                'agreement_level': confluence_result['agreement_level']
            }
            
            # Add price data from primary timeframe
            if not primary_df.empty:
                close_col = 'close' if 'close' in primary_df.columns else 'Close'
                if close_col in primary_df.columns:
                    signal['current_price'] = float(primary_df[close_col].iloc[-1])
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe signal detection: {e}")
            return None