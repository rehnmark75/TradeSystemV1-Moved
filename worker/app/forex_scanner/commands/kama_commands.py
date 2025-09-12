# commands/kama_commands.py
"""
KAMA-specific command implementations - FIXED VERSION
Handles KAMA testing, analysis, and performance evaluation
Fixed to use correct DataFetcher method names
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from core.strategies.kama_strategy import KAMAStrategy
    from core.strategies.ema_strategy import EMAStrategy
    from analysis.technical import TechnicalAnalyzer
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.strategies.kama_strategy import KAMAStrategy
    from forex_scanner.core.strategies.ema_strategy import EMAStrategy
    from forex_scanner.analysis.technical import TechnicalAnalyzer
    from forex_scanner import config


class KAMACommands:
    """Command implementations for KAMA-specific operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager)
        self.technical_analyzer = TechnicalAnalyzer()
    
    def test_kama_calculation(
        self, 
        pair: str, 
        timeframe: str = '5m', 
        days: int = 7,
        kama_config: str = 'default'
    ):
        """
        Test KAMA indicator calculation and display results
        
        Args:
            pair: Trading pair to test
            timeframe: Timeframe for testing
            days: Number of days to analyze
            kama_config: KAMA configuration to use
        """
        self.logger.info(f"🔄 Testing KAMA calculation for {pair}")
        self.logger.info(f"   Timeframe: {timeframe}, Days: {days}, Config: {kama_config}")
        
        try:
            # FIXED: Use correct method name and parameters
            lookback_hours = days * 24
            df = self.data_fetcher.get_enhanced_data(
                epic=pair,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df is None or len(df) < 50:
                self.logger.error(f"❌ Insufficient data for {pair}: {len(df) if df is not None else 0} bars")
                return False
            
            # Get KAMA configuration
            kama_config_data = config.KAMA_STRATEGY_CONFIG.get(kama_config, config.KAMA_STRATEGY_CONFIG['default'])
            period = kama_config_data['period']
            fast = kama_config_data['fast']
            slow = kama_config_data['slow']
            
            self.logger.info(f"📊 KAMA Parameters: Period={period}, Fast={fast}, Slow={slow}")
            
            # Calculate KAMA if not already present
            df_with_kama = self._ensure_kama_indicators(df, period, fast, slow)
            
            # Display recent KAMA values
            recent_data = df_with_kama.tail(10)
            
            self.logger.info("📈 Recent KAMA Values:")
            self.logger.info("   Time                 | Close    | KAMA     | ER      | Trend | Signal")
            self.logger.info("   --------------------|----------|----------|---------|-------|--------")
            
            for idx, row in recent_data.iterrows():
                time_str = row['start_time'].strftime('%Y-%m-%d %H:%M') if 'start_time' in row else str(idx)
                close = row['close']
                kama = row.get(f'kama_{period}', row.get('kama', 0))
                er = row.get(f'kama_{period}_er', row.get('kama_er', 0))
                trend = row.get(f'kama_{period}_trend', row.get('kama_trend', 0))
                signal = row.get(f'kama_{period}_signal', row.get('kama_signal', 0))
                
                trend_arrow = '↗️' if trend > 0 else '↘️' if trend < 0 else '→'
                signal_indicator = '🔥' if abs(signal) > 1 else '  '
                
                self.logger.info(f"   {time_str} | {close:.5f} | {kama:.5f} | {er:.3f} | {trend_arrow}   | {signal_indicator}")
            
            # Calculate KAMA statistics
            self._display_kama_statistics(df_with_kama, kama_config, period)
            
            # Test KAMA strategy signals
            self._test_kama_signals(df_with_kama, pair, kama_config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error testing KAMA calculation: {e}")
            return False
    
    def analyze_kama_performance(
        self, 
        pair: str, 
        timeframe: str = '5m', 
        days: int = 30
    ):
        """
        Analyze KAMA performance metrics and efficiency ratios
        
        Args:
            pair: Trading pair to analyze
            timeframe: Timeframe for analysis
            days: Number of days to analyze
        """
        self.logger.info(f"📊 Analyzing KAMA performance for {pair}")
        
        try:
            # FIXED: Use correct method name
            lookback_hours = days * 24
            df = self.data_fetcher.get_enhanced_data(
                epic=pair,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df is None or len(df) < 100:
                self.logger.error(f"❌ Insufficient data for analysis")
                return False
            
            # Analyze efficiency ratio patterns
            self._analyze_efficiency_ratios(df, pair)
            
            # Analyze trend detection accuracy
            self._analyze_trend_detection(df, pair)
            
            # Analyze signal quality
            self._analyze_signal_quality(df, pair)
            
            # Market regime analysis
            self._analyze_market_regimes(df, pair)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing KAMA performance: {e}")
            return False
    
    def compare_kama_vs_ema(
        self, 
        pair: str, 
        timeframe: str = '5m', 
        days: int = 30
    ):
        """
        Compare KAMA vs EMA performance
        
        Args:
            pair: Trading pair to compare
            timeframe: Timeframe for comparison
            days: Number of days to compare
        """
        self.logger.info(f"⚖️ Comparing KAMA vs EMA for {pair}")
        
        try:
            # FIXED: Use correct method name
            lookback_hours = days * 24
            df = self.data_fetcher.get_enhanced_data(
                epic=pair,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df is None or len(df) < 100:
                self.logger.error(f"❌ Insufficient data for comparison")
                return False
            
            # Initialize strategies
            kama_strategy = KAMAStrategy()
            ema_strategy = EMAStrategy(data_fetcher=self.data_fetcher)
            
            # Get signals from both strategies
            kama_signals = self._get_strategy_signals(df, kama_strategy, pair, timeframe)
            ema_signals = self._get_strategy_signals(df, ema_strategy, pair, timeframe)
            
            # Compare performance metrics
            self._compare_strategy_performance(kama_signals, ema_signals, 'KAMA', 'EMA')
            
            # Compare noise reduction
            self._compare_noise_reduction(df, pair)
            
            # Compare trend following accuracy
            self._compare_trend_accuracy(df, kama_signals, ema_signals)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing KAMA vs EMA: {e}")
            return False
    
    def show_recent_signals(
        self, 
        pair: str, 
        timeframe: str = '5m', 
        days: int = 14,
        min_confidence: float = 0.6
    ):
        """
        Show recent KAMA signals for a trading pair
        
        Args:
            pair: Trading pair to analyze
            timeframe: Timeframe for signals
            days: Number of days to look back
            min_confidence: Minimum confidence threshold
        """
        self.logger.info(f"🎯 Recent KAMA signals for {pair}")
        self.logger.info(f"   Period: {days} days, Min confidence: {min_confidence:.1%}")
        
        try:
            # FIXED: Use correct method name
            lookback_hours = days * 24
            df = self.data_fetcher.get_enhanced_data(
                epic=pair,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df is None or len(df) < 50:
                self.logger.error(f"❌ Insufficient data for {pair}")
                return False
            
            # Get KAMA signals
            kama_strategy = KAMAStrategy()
            signals = []
            
            # Check each bar for signals
            for i in range(50, len(df)):  # Need enough data for KAMA calculation
                try:
                    df_slice = df.iloc[:i+1].copy()
                    signal = kama_strategy.detect_signal(df_slice, pair, config.SPREAD_PIPS, timeframe)
                    
                    if signal and signal['confidence_score'] >= min_confidence:
                        signals.append(signal)
                        
                except Exception:
                    continue
            
            # Display signals
            if signals:
                self.logger.info(f"📊 Found {len(signals)} KAMA signals above {min_confidence:.1%} confidence:")
                self.logger.info("   Time                 | Type | Conf% | Price    | ER    | Trigger")
                self.logger.info("   --------------------|------|-------|----------|-------|------------------")
                
                for signal in signals[-20:]:  # Show last 20 signals
                    timestamp = signal['timestamp']
                    signal_type = signal['signal_type']
                    confidence = signal['confidence_score']
                    price = signal['signal_price']
                    er = signal.get('efficiency_ratio', 0)
                    trigger = signal.get('trigger_reason', '')[:15] + '...' if len(signal.get('trigger_reason', '')) > 15 else signal.get('trigger_reason', '')
                    
                    type_icon = '🟢' if signal_type == 'BULL' else '🔴'
                    
                    self.logger.info(f"   {timestamp} | {type_icon}   | {confidence:.1%} | {price:.5f} | {er:.3f} | {trigger}")
            else:
                self.logger.info(f"📊 No KAMA signals found above {min_confidence:.1%} confidence")
            
            return True
                
        except Exception as e:
            self.logger.error(f"❌ Error showing recent signals: {e}")
            return False
    
    def _ensure_kama_indicators(self, df: pd.DataFrame, period: int, fast: int, slow: int) -> pd.DataFrame:
        """Ensure KAMA indicators are present in dataframe"""
        try:
            kama_col = f'kama_{period}'
            er_col = f'kama_{period}_er'
            
            if kama_col not in df.columns or er_col not in df.columns:
                self.logger.info("🔄 Calculating KAMA indicators...")
                df_with_kama = self.technical_analyzer.calculate_kama(
                    df, period=period, fast_sc=fast, slow_sc=slow
                )
                return df_with_kama
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error ensuring KAMA indicators: {e}")
            return df
    
    def _display_kama_statistics(self, df: pd.DataFrame, kama_config: str, period: int):
        """Display KAMA calculation statistics"""
        try:
            kama_col = f'kama_{period}'
            er_col = f'kama_{period}_er'
            signal_col = f'kama_{period}_signal'
            
            # Try multiple column name formats for compatibility
            if kama_col not in df.columns:
                kama_col = 'kama'
            if er_col not in df.columns:
                er_col = 'kama_er'
            if signal_col not in df.columns:
                signal_col = 'kama_signal'
            
            kama_values = df[kama_col].dropna() if kama_col in df.columns else pd.Series()
            er_values = df[er_col].dropna() if er_col in df.columns else pd.Series()
            
            if len(kama_values) == 0:
                self.logger.warning("⚠️ No valid KAMA values found")
                return
            
            self.logger.info(f"📊 KAMA Statistics ({kama_config} config):")
            self.logger.info(f"   Valid KAMA values: {len(kama_values)}")
            self.logger.info(f"   Efficiency Ratio avg: {er_values.mean():.3f}")
            self.logger.info(f"   Efficiency Ratio std: {er_values.std():.3f}")
            self.logger.info(f"   ER > 0.5 (trending): {(er_values > 0.5).sum()}/{len(er_values)} ({(er_values > 0.5).mean():.1%})")
            self.logger.info(f"   ER < 0.3 (choppy): {(er_values < 0.3).sum()}/{len(er_values)} ({(er_values < 0.3).mean():.1%})")
            
            # Trend change frequency
            if signal_col in df.columns:
                trend_changes = df[signal_col].abs().sum()
                self.logger.info(f"   Trend changes: {trend_changes} over {len(df)} bars")
                self.logger.info(f"   Change frequency: {trend_changes / len(df) * 100:.1f} per 100 bars")
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating KAMA statistics: {e}")
    
    def _test_kama_signals(self, df: pd.DataFrame, pair: str, kama_config: str):
        """Test KAMA signal generation"""
        try:
            kama_strategy = KAMAStrategy(kama_config)
            
            # Test signal detection on recent data
            signal = kama_strategy.detect_signal(df, pair, config.SPREAD_PIPS, '5m')
            
            self.logger.info("🎯 Current KAMA Signal Test:")
            if signal:
                self.logger.info(f"   Signal detected: {signal['signal_type']}")
                self.logger.info(f"   Confidence: {signal['confidence_score']:.1%}")
                self.logger.info(f"   Trigger: {signal['trigger_reason']}")
                self.logger.info(f"   Price: {signal['signal_price']:.5f}")
                self.logger.info(f"   Efficiency Ratio: {signal.get('efficiency_ratio', 0):.3f}")
            else:
                self.logger.info("   No signal detected at current time")
                
            # Test data validation
            validation = kama_strategy._validate_kama_data(df)
            self.logger.info(f"   Data validation: {'✅ Passed' if validation else '❌ Failed'}")
            
        except Exception as e:
            self.logger.error(f"❌ Error testing KAMA signals: {e}")
    
    def _analyze_efficiency_ratios(self, df: pd.DataFrame, pair: str):
        """Analyze efficiency ratio patterns"""
        try:
            # Try multiple column name formats
            er_cols = ['kama_10_er', 'kama_er', 'kama_14_er', 'kama_8_er']
            er_col = None
            
            for col in er_cols:
                if col in df.columns:
                    er_col = col
                    break
            
            if er_col is None:
                self.logger.warning("⚠️ KAMA efficiency ratio not found in data")
                return
            
            er_values = df[er_col].dropna()
            if len(er_values) == 0:
                self.logger.warning("⚠️ No valid efficiency ratio values")
                return
            
            self.logger.info(f"📊 Efficiency Ratio Analysis for {pair}:")
            self.logger.info(f"   Mean ER: {er_values.mean():.3f}")
            self.logger.info(f"   Std ER: {er_values.std():.3f}")
            self.logger.info(f"   High trending (ER > 0.6): {(er_values > 0.6).mean():.1%}")
            self.logger.info(f"   Moderate trending (ER 0.3-0.6): {((er_values >= 0.3) & (er_values <= 0.6)).mean():.1%}")
            self.logger.info(f"   Choppy market (ER < 0.3): {(er_values < 0.3).mean():.1%}")
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing efficiency ratios: {e}")
    
    def _analyze_trend_detection(self, df: pd.DataFrame, pair: str):
        """Analyze trend detection accuracy"""
        try:
            # Look for trend columns
            trend_cols = ['kama_10_trend', 'kama_trend', 'kama_14_trend', 'kama_8_trend']
            trend_col = None
            
            for col in trend_cols:
                if col in df.columns:
                    trend_col = col
                    break
            
            if trend_col is None:
                self.logger.warning("⚠️ KAMA trend data not found")
                return
            
            trend_values = df[trend_col].dropna()
            if len(trend_values) == 0:
                return
            
            bullish_periods = (trend_values > 0).sum()
            bearish_periods = (trend_values < 0).sum()
            neutral_periods = (trend_values == 0).sum()
            
            self.logger.info(f"📈 Trend Detection for {pair}:")
            self.logger.info(f"   Bullish periods: {bullish_periods}/{len(trend_values)} ({bullish_periods/len(trend_values):.1%})")
            self.logger.info(f"   Bearish periods: {bearish_periods}/{len(trend_values)} ({bearish_periods/len(trend_values):.1%})")
            self.logger.info(f"   Neutral periods: {neutral_periods}/{len(trend_values)} ({neutral_periods/len(trend_values):.1%})")
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing trend detection: {e}")
    
    def _analyze_signal_quality(self, df: pd.DataFrame, pair: str):
        """Analyze signal quality"""
        try:
            # Look for signal columns
            signal_cols = ['kama_10_signal', 'kama_signal', 'kama_14_signal', 'kama_8_signal']
            signal_col = None
            
            for col in signal_cols:
                if col in df.columns:
                    signal_col = col
                    break
            
            if signal_col is None:
                self.logger.warning("⚠️ KAMA signal data not found")
                return
            
            signal_values = df[signal_col].dropna()
            if len(signal_values) == 0:
                return
            
            strong_bull_signals = (signal_values == 2).sum()
            weak_bull_signals = (signal_values == 1).sum()
            strong_bear_signals = (signal_values == -2).sum()
            weak_bear_signals = (signal_values == -1).sum()
            no_signals = (signal_values == 0).sum()
            
            total_signals = strong_bull_signals + weak_bull_signals + strong_bear_signals + weak_bear_signals
            
            self.logger.info(f"🎯 Signal Quality for {pair}:")
            self.logger.info(f"   Total signals: {total_signals}/{len(signal_values)} ({total_signals/len(signal_values):.1%})")
            self.logger.info(f"   Strong bull: {strong_bull_signals} ({strong_bull_signals/len(signal_values):.1%})")
            self.logger.info(f"   Weak bull: {weak_bull_signals} ({weak_bull_signals/len(signal_values):.1%})")
            self.logger.info(f"   Strong bear: {strong_bear_signals} ({strong_bear_signals/len(signal_values):.1%})")
            self.logger.info(f"   Weak bear: {weak_bear_signals} ({weak_bear_signals/len(signal_values):.1%})")
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing signal quality: {e}")
    
    def _analyze_market_regimes(self, df: pd.DataFrame, pair: str):
        """Analyze market regime analysis"""
        try:
            if 'close' not in df.columns:
                return
            
            # Calculate price volatility
            df['price_change'] = df['close'].pct_change()
            volatility = df['price_change'].std()
            
            # Calculate trend strength
            if len(df) > 20:
                recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            else:
                recent_trend = 0
            
            self.logger.info(f"🌡️ Market Regime Analysis for {pair}:")
            self.logger.info(f"   Recent volatility: {volatility:.4f}")
            self.logger.info(f"   20-period trend: {recent_trend:.2%}")
            
            if volatility > 0.01:
                regime = "High Volatility"
            elif volatility > 0.005:
                regime = "Medium Volatility"
            else:
                regime = "Low Volatility"
            
            self.logger.info(f"   Market regime: {regime}")
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing market regimes: {e}")
    
    def _get_strategy_signals(self, df: pd.DataFrame, strategy, pair: str, timeframe: str) -> List[Dict]:
        """Get signals from a strategy"""
        signals = []
        try:
            # Test strategy at multiple points
            for i in range(50, len(df), 10):  # Sample every 10 bars
                try:
                    df_slice = df.iloc[:i+1].copy()
                    signal = strategy.detect_signal(df_slice, pair, config.SPREAD_PIPS, timeframe)
                    if signal:
                        signals.append(signal)
                except Exception:
                    continue
        except Exception as e:
            self.logger.error(f"❌ Error getting strategy signals: {e}")
        
        return signals
    
    def _compare_strategy_performance(self, signals1: List[Dict], signals2: List[Dict], name1: str, name2: str):
        """Compare performance between two strategy signal sets"""
        try:
            self.logger.info(f"⚖️ Strategy Comparison: {name1} vs {name2}")
            
            # Signal counts
            self.logger.info(f"   {name1} signals: {len(signals1)}")
            self.logger.info(f"   {name2} signals: {len(signals2)}")
            
            if not signals1 or not signals2:
                self.logger.info("   Cannot compare - insufficient signals")
                return
            
            # Confidence comparison
            conf1 = [s['confidence_score'] for s in signals1]
            conf2 = [s['confidence_score'] for s in signals2]
            
            self.logger.info(f"   {name1} avg confidence: {np.mean(conf1):.1%}")
            self.logger.info(f"   {name2} avg confidence: {np.mean(conf2):.1%}")
            
            # High confidence signals
            high_conf1 = sum(1 for c in conf1 if c > 0.8)
            high_conf2 = sum(1 for c in conf2 if c > 0.8)
            
            self.logger.info(f"   {name1} high confidence: {high_conf1}/{len(signals1)} ({high_conf1/len(signals1):.1%})")
            self.logger.info(f"   {name2} high confidence: {high_conf2}/{len(signals2)} ({high_conf2/len(signals2):.1%})")
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing strategy performance: {e}")
    
    def _compare_noise_reduction(self, df: pd.DataFrame, pair: str):
        """Compare noise reduction between KAMA and EMA"""
        try:
            # Look for both KAMA and EMA columns
            kama_cols = ['kama_10', 'kama', 'kama_14', 'kama_8']
            ema_cols = ['ema_21', 'ema_20', 'ema_9']
            
            kama_col = None
            ema_col = None
            
            for col in kama_cols:
                if col in df.columns:
                    kama_col = col
                    break
            
            for col in ema_cols:
                if col in df.columns:
                    ema_col = col
                    break
            
            if kama_col is None or ema_col is None:
                self.logger.warning("⚠️ Required indicators not found for noise comparison")
                return
            
            kama_values = df[kama_col].dropna()
            ema_values = df[ema_col].dropna()
            close_values = df['close'].dropna()
            
            if len(kama_values) == 0 or len(ema_values) == 0:
                return
            
            # Calculate smoothness (less variation = smoother)
            kama_smoothness = 1 / (kama_values.diff().abs().mean() + 0.0001)
            ema_smoothness = 1 / (ema_values.diff().abs().mean() + 0.0001)
            
            self.logger.info(f"🔇 Noise Reduction Comparison for {pair}:")
            self.logger.info(f"   KAMA smoothness: {kama_smoothness:.2f}")
            self.logger.info(f"   EMA smoothness: {ema_smoothness:.2f}")
            self.logger.info(f"   KAMA advantage: {((kama_smoothness - ema_smoothness) / ema_smoothness * 100):+.1f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing noise reduction: {e}")
    
    def _compare_trend_accuracy(self, df: pd.DataFrame, kama_signals: List[Dict], ema_signals: List[Dict]):
        """Compare trend following accuracy"""
        try:
            if not kama_signals or not ema_signals:
                self.logger.info("⚠️ Insufficient signals for trend accuracy comparison")
                return
            
            # Simple comparison based on signal timing and direction
            self.logger.info("📊 Trend Following Comparison:")
            self.logger.info(f"   KAMA signals: {len(kama_signals)}")
            self.logger.info(f"   EMA signals: {len(ema_signals)}")
            
            # Count signal types
            kama_bull = sum(1 for s in kama_signals if s['signal_type'] == 'BULL')
            kama_bear = sum(1 for s in kama_signals if s['signal_type'] == 'BEAR')
            ema_bull = sum(1 for s in ema_signals if s['signal_type'] == 'BULL')
            ema_bear = sum(1 for s in ema_signals if s['signal_type'] == 'BEAR')
            
            self.logger.info(f"   KAMA: {kama_bull} BULL, {kama_bear} BEAR")
            self.logger.info(f"   EMA: {ema_bull} BULL, {ema_bear} BEAR")
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing trend accuracy: {e}")
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 5)