# core/signal_detector.py
"""
Signal Detection Coordinator
Lightweight coordinator that delegates to specialized strategy modules

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategies have been archived to forex_scanner/archive/disabled_strategies/
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

try:
    from .database import DatabaseManager
    from .data_fetcher import DataFetcher
    from .backtest.performance_analyzer import PerformanceAnalyzer
    from .backtest.signal_analyzer import SignalAnalyzer
    from .detection.price_adjuster import PriceAdjuster
    from .smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    # NOTE: LargeCandleFilter removed (Jan 2026) - database columns dropped
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer
    from forex_scanner.core.detection.price_adjuster import PriceAdjuster
    from forex_scanner.core.smart_money_integration import add_smart_money_to_signal, add_smart_money_to_signals
    # NOTE: LargeCandleFilter removed (Jan 2026) - database columns dropped

# Database-driven configuration services (NOT legacy config files)
try:
    from services.scanner_config_service import get_scanner_config
    from services.smc_simple_config_service import get_smc_simple_config
except ImportError:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    from forex_scanner.services.smc_simple_config_service import get_smc_simple_config


class SignalDetector:
    """
    Lightweight signal detection coordinator

    After January 2026 cleanup, only SMC Simple strategy is active.
    Legacy strategies are archived in forex_scanner/archive/disabled_strategies/
    """

    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm', config_override: dict = None):
        self.db_manager = db_manager
        self.data_fetcher = DataFetcher(db_manager, user_timezone)
        self.price_adjuster = PriceAdjuster()
        self.logger = logging.getLogger(__name__)

        # Store config override for backtest parameter isolation
        self._config_override = config_override

        # NOTE: Large candle filter removed (Jan 2026 cleanup)
        # The filter was initialized but never called in the signal flow.
        # Database columns for large candle filter were dropped.
        # See migration: remove_safety_filter_columns.sql
        self.large_candle_filter = None

        # Initialize SMC Simple Strategy (the only active strategy)
        # NOTE: SMC Simple is always enabled after January 2026 cleanup
        # Configuration is loaded from database via get_smc_simple_config()
        try:
            # SMC Simple strategy uses lazy loading for consistency
            self.smc_simple_enabled = True
            self.smc_simple_strategy = None  # Will be lazy-loaded on first use
            self.logger.info("✅ SMC Simple strategy enabled (3-tier EMA, lazy-load, database config)")
        except Exception as e:
            self.logger.error(f"❌ Failed to enable SMC Simple strategy: {e}")
            self.smc_simple_enabled = False
            self.smc_simple_strategy = None

        # Initialize analysis components
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

        self.logger.info("📊 SignalDetector initialized (SMC Simple only - legacy strategies archived)")

    # =========================================================================
    # BACKTEST FORCE-INITIALIZATION METHODS
    # =========================================================================

    def force_initialize_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Force-initialize a specific strategy for backtesting, regardless of config flags.

        After January 2026 cleanup, only SMC Simple strategy is available.
        Other strategy names are kept for backward compatibility but will fail.

        Args:
            strategy_name: Strategy name (e.g., 'SMC_SIMPLE', 'SMC')

        Returns:
            Tuple of (success: bool, message: str)
        """
        strategy_name = strategy_name.upper()

        # Strategy initialization mapping - only SMC Simple is active
        init_map = {
            'SMC_SIMPLE': self._force_init_smc_simple,
            'SMC_EMA': self._force_init_smc_simple,
            'SMC': self._force_init_smc_simple,  # Default SMC to SMC_SIMPLE
        }

        if strategy_name not in init_map:
            return False, f"Unknown or archived strategy: {strategy_name}. Only SMC_SIMPLE is available after cleanup."

        return init_map[strategy_name]()

    def _force_init_smc_simple(self) -> Tuple[bool, str]:
        """Force-initialize SMC Simple strategy for backtest"""
        try:
            # SMC Simple uses lazy loading, just enable the flag
            self.smc_simple_enabled = True
            self.smc_simple_strategy = None  # Will be lazy-loaded on first use
            self.logger.info("🔧 Force-initialized SMC Simple strategy (lazy-load)")
            return True, "SMC Simple strategy force-initialized"
        except Exception as e:
            return False, f"Failed to force-init SMC Simple: {e}"

    def _get_default_timeframe(self, timeframe: str = None) -> str:
        """Get default timeframe from database config if not specified"""
        if timeframe is None:
            scanner_config = get_scanner_config()
            return scanner_config.default_timeframe or '15m'
        return timeframe

    # =========================================================================
    # SIGNAL DETECTION METHODS
    # =========================================================================

    def detect_smc_simple_signals(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> Optional[Dict]:
        """
        Detect SMC Simple 3-tier signals (50 EMA based)

        Uses multi-timeframe analysis:
        - Bias: 4H 50 EMA for directional bias
        - Trigger: 15m swing break with body-close confirmation
        - Entry: 5m pullback to Fibonacci zone
        """
        # Load timeframes from DATABASE config (not legacy config files)
        smc_config = get_smc_simple_config()
        htf_tf = smc_config.htf_timeframe or '4h'
        trigger_tf = smc_config.trigger_timeframe or '15m'
        entry_tf = smc_config.entry_timeframe or '5m'

        # Check if epic is enabled in pair overrides (database)
        # get_smc_simple_config() already handles enabled pairs via pair_overrides table

        self.logger.debug(f"🔍 [SMC_SIMPLE] Using htf_tf={htf_tf}, trigger_tf={trigger_tf}, entry_tf={entry_tf}")

        try:
            # Initialize strategy if not already done (lazy loading)
            if not hasattr(self, 'smc_simple_strategy') or self.smc_simple_strategy is None:
                from .strategies.smc_simple_strategy import create_smc_simple_strategy

                # Strategy loads all config from database via smc_simple_config_service
                # Pass config_override for backtest parameter isolation
                self.smc_simple_strategy = create_smc_simple_strategy(
                    None,
                    logger=self.logger,
                    db_manager=self.db_manager,
                    config_override=self._config_override
                )
                mode_str = "BACKTEST MODE with overrides" if self._config_override else "LIVE MODE"
                self.logger.info(f"✅ SMC Simple strategy initialized ({mode_str}, htf={htf_tf}, trigger={trigger_tf}, entry={entry_tf})")

            # Check if data_fetcher is in backtest mode (needed for lookback calculations)
            is_backtest = hasattr(self.data_fetcher, 'current_backtest_time') and self.data_fetcher.current_backtest_time is not None

            # CRITICAL FIX: Reset cooldowns at start of new backtest to prevent stale cooldowns
            if is_backtest and hasattr(self.smc_simple_strategy, 'reset_cooldowns'):
                current_backtest_id = id(self.data_fetcher)
                if not hasattr(self, '_smc_simple_backtest_id') or self._smc_simple_backtest_id != current_backtest_id:
                    self._smc_simple_backtest_id = current_backtest_id
                    self.smc_simple_strategy.reset_cooldowns()

            # Get 4H data for EMA bias
            htf_lookback = 4000 if is_backtest else 400
            df_4h = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=htf_tf,
                lookback_hours=htf_lookback
            )

            if df_4h is None or len(df_4h) < 60:
                self.logger.debug(f"Insufficient {htf_tf} data for {epic} (got {len(df_4h) if df_4h is not None else 0} bars)")
                return None

            # Get trigger timeframe data for swing break detection
            if trigger_tf == '15m':
                trigger_lookback = 2000 if is_backtest else 30
            else:
                trigger_lookback = 2000 if is_backtest else 100

            df_trigger = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=trigger_tf,
                lookback_hours=trigger_lookback
            )

            if df_trigger is None or len(df_trigger) < 30:
                self.logger.debug(f"Insufficient {trigger_tf} data for {epic} (got {len(df_trigger) if df_trigger is not None else 0} bars)")
                return None

            # Get entry timeframe data for pullback entry
            df_entry = None
            if entry_tf in ['15m', '5m']:
                if entry_tf == '5m':
                    entry_lookback = 2000 if is_backtest else 25
                else:
                    entry_lookback = 2000 if is_backtest else 50

                df_entry = self.data_fetcher.get_enhanced_data(
                    epic=epic,
                    pair=pair,
                    timeframe=entry_tf,
                    lookback_hours=entry_lookback
                )

            df_entry_len = len(df_entry) if df_entry is not None else 0
            self.logger.info(f"🔍 [SMC_SIMPLE] Passing to strategy: 4H({len(df_4h)} bars), {trigger_tf}({len(df_trigger)} bars), {entry_tf}({df_entry_len} bars)")

            # Detect signal
            signal = self.smc_simple_strategy.detect_signal(
                df_trigger=df_trigger,
                df_4h=df_4h,
                epic=epic,
                pair=pair,
                df_entry=df_entry
            )

            if signal:
                self.logger.info(f"✅ [SMC_SIMPLE] Signal detected for {epic}: {signal['signal']} @ {signal['entry_price']:.5f}")
                signal = self._add_market_context(signal, df_trigger)

            # Flush any pending rejections to database
            if hasattr(self, 'smc_simple_strategy') and self.smc_simple_strategy is not None:
                self.smc_simple_strategy.flush_rejections()

            return signal

        except Exception as e:
            self.logger.error(f"❌ [SMC_SIMPLE] Error detecting signals for {epic}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def detect_signals_all_strategies(
        self,
        epic: str,
        pair: str,
        spread_pips: float = 1.5,
        timeframe: str = None
    ) -> List[Dict]:
        """
        Detect signals using all enabled strategies.

        After January 2026 cleanup, only SMC Simple strategy is active.
        """
        all_signals = []

        try:
            self.logger.debug(f"🔍 Running strategy detection for {epic}")

            individual_results = {}

            # SMC Simple Strategy (the only active strategy after January 2026 cleanup)
            # SMC Simple is always enabled - no config flag needed
            if self.smc_simple_enabled:
                try:
                    self.logger.debug(f"🔍 [SMC SIMPLE] Starting detection for {epic}")
                    smc_simple_signal = self.detect_smc_simple_signals(epic, pair, spread_pips, timeframe)

                    individual_results['smc_simple'] = smc_simple_signal

                    if smc_simple_signal:
                        all_signals.append(smc_simple_signal)
                        self.logger.info(f"✅ [SMC SIMPLE] Signal detected for {epic}: {smc_simple_signal.get('signal')} @ {smc_simple_signal.get('entry_price', 0):.5f}")
                    else:
                        self.logger.debug(f"📊 [SMC SIMPLE] No signal for {epic}")

                except Exception as e:
                    self.logger.error(f"❌ [SMC SIMPLE] Error for {epic}: {e}")
                    individual_results['smc_simple'] = None

            # Add smart money analysis to all signals (if enabled)
            if all_signals:
                try:
                    all_signals = add_smart_money_to_signals(all_signals, self.data_fetcher, self.db_manager)
                    self.logger.debug(f"✅ Smart money analysis added to {len(all_signals)} signals")
                except Exception as e:
                    self.logger.warning(f"⚠️ Smart money analysis failed: {e}")

            # Results logging
            if all_signals:
                strategy_names = [s.get('strategy', 'unknown') for s in all_signals]
                self.logger.info(f"🎯 {epic}: {len(all_signals)} signals from strategies: {', '.join(strategy_names)}")

                for i, signal in enumerate(all_signals, 1):
                    strategy = signal.get('strategy', 'unknown')
                    signal_type = signal.get('signal_type', 'unknown')
                    confidence = signal.get('confidence_score', 0)
                    self.logger.info(f"   📈 Signal {i}: {strategy} - {signal_type} ({confidence:.1%})")
            else:
                self.logger.debug(f"📊 {epic}: No signals from any strategy")

            return all_signals

        except Exception as e:
            self.logger.error(f"❌ Error in detect_signals_all_strategies for {epic}: {e}")
            return all_signals

    def detect_signals(self, epic: str, pair: str, spread_pips: float = 1.5, timeframe: str = None) -> List[Dict]:
        """
        Main signal detection entry point.
        Delegates to detect_signals_all_strategies.
        """
        return self.detect_signals_all_strategies(epic, pair, spread_pips, timeframe)

    # =========================================================================
    # MARKET CONTEXT AND ENHANCEMENT METHODS
    # =========================================================================

    def _add_market_context(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """ENHANCED: Add comprehensive market context including complete technical indicators"""

        # First, add complete technical indicators
        signal = self._add_complete_technical_indicators(signal, df)

        # Then add existing market context
        if df is None or df.empty:
            return signal

        try:
            latest = df.iloc[-1]

            # Add available market context
            context_fields = [
                'volume_ratio_20', 'distance_to_support_pips', 'distance_to_resistance_pips',
                'trend_alignment', 'consolidation_range_pips', 'bars_since_breakout',
                'rejection_wicks_count', 'consecutive_green_candles', 'consecutive_red_candles'
            ]

            for field in context_fields:
                if field in latest.index:
                    signal[field] = latest[field]

            # Add volume confirmation flag
            if 'volume' in signal and 'volume_sma_20' in latest.index:
                signal['volume_confirmation'] = signal['volume'] > latest['volume_sma_20'] * 1.2

            # Add recent price action summary
            if len(df) >= 5:
                recent_data = df.tail(5)
                signal['recent_price_action'] = {
                    'bars_count': len(recent_data),
                    'high_range': float(recent_data['high'].max()),
                    'low_range': float(recent_data['low'].min()),
                    'avg_volume': float(recent_data.get('ltv', recent_data.get('volume', pd.Series([0]))).mean()),
                    'price_trend': 'bullish' if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else 'bearish'
                }

            self.logger.debug(f"✅ Enhanced signal with complete market context for {signal.get('epic')}")
            return signal

        except Exception as e:
            self.logger.error(f"❌ Error adding enhanced market context: {e}")
            return signal

    def _add_complete_technical_indicators(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """
        Add complete technical indicators from DataFrame to signal.
        This ensures Claude gets ALL available technical data for comprehensive analysis.
        """
        if df is None or df.empty:
            return signal

        try:
            latest = df.iloc[-1]

            # 1. PRICE DATA
            signal.update({
                'current_price': float(latest['close']),
                'open_price': float(latest['open']),
                'high_price': float(latest['high']),
                'low_price': float(latest['low']),
                'close_price': float(latest['close'])
            })

            # 2. EMA INDICATORS
            ema_indicators = {}
            for col in df.columns:
                if col.startswith('ema_') and col.replace('ema_', '').isdigit():
                    try:
                        period = int(col.replace('ema_', ''))
                        ema_indicators[col] = float(latest[col])

                        if period == 9:
                            signal['ema_9'] = float(latest[col])
                            signal['ema_short'] = float(latest[col])
                        elif period == 21:
                            signal['ema_21'] = float(latest[col])
                            signal['ema_long'] = float(latest[col])
                        elif period == 200:
                            signal['ema_200'] = float(latest[col])
                            signal['ema_trend'] = float(latest[col])

                    except (ValueError, KeyError):
                        continue

            if ema_indicators:
                signal.update(ema_indicators)

            # 3. MACD INDICATORS
            macd_indicators = {}
            macd_mappings = {
                'macd_line': ['macd_line', 'macd', 'macd_12_26_9'],
                'macd_signal': ['macd_signal', 'macd_signal_line', 'macd_signal_12_26_9'],
                'macd_histogram': ['macd_histogram', 'macd_hist', 'macd_histogram_12_26_9']
            }

            for standard_name, possible_cols in macd_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            macd_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 4. KAMA INDICATORS
            kama_indicators = {}
            kama_mappings = {
                'kama_value': ['kama_value', 'kama', 'kama_10', 'kama_14'],
                'efficiency_ratio': ['efficiency_ratio', 'kama_er', 'kama_10_er', 'kama_14_er'],
                'kama_trend': ['kama_trend', 'kama_slope', 'kama_10_trend', 'kama_14_trend']
            }

            for standard_name, possible_cols in kama_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            kama_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 5. ADX AND DIRECTIONAL INDICATORS
            adx_indicators = {}
            adx_mappings = {
                'adx': ['adx', 'adx_14'],
                'plus_di': ['plus_di', 'di_plus', 'plus_di_14', '+di'],
                'minus_di': ['minus_di', 'di_minus', 'minus_di_14', '-di'],
                'dx': ['dx', 'dx_14']
            }

            for standard_name, possible_cols in adx_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            adx_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 6. OTHER TECHNICAL INDICATORS
            other_indicators = {}
            other_mappings = {
                'rsi': ['rsi', 'rsi_14'],
                'atr': ['atr', 'atr_14'],
                'bb_upper': ['bb_upper', 'bollinger_upper', 'bb_upper_20_2'],
                'bb_middle': ['bb_middle', 'bollinger_middle', 'bb_middle_20_2'],
                'bb_lower': ['bb_lower', 'bollinger_lower', 'bb_lower_20_2']
            }

            for standard_name, possible_cols in other_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            other_indicators[standard_name] = float(latest[col])
                            signal[standard_name] = float(latest[col])
                            break
                        except (ValueError, KeyError):
                            continue

            # 7. VOLUME DATA
            volume_fields = ['ltv', 'volume', 'volume_sma_20', 'volume_ratio_20']
            for field in volume_fields:
                if field in df.columns:
                    try:
                        if field == 'ltv' or field == 'volume':
                            signal['volume'] = float(latest[field])
                        else:
                            signal[field] = float(latest[field])
                    except (ValueError, KeyError):
                        continue

            # 8. SWING POINT DATA
            swing_data = {}
            swing_fields = {
                'swing_high': ['swing_high', 'recent_swing_high', 'swing_high_price'],
                'swing_low': ['swing_low', 'recent_swing_low', 'swing_low_price'],
                'distance_to_swing_high_pips': ['distance_to_swing_high_pips', 'swing_high_distance_pips'],
                'distance_to_swing_low_pips': ['distance_to_swing_low_pips', 'swing_low_distance_pips'],
                'nearest_swing_type': ['nearest_swing_type', 'swing_type'],
                'swing_strength': ['swing_strength', 'swing_level_strength']
            }

            for standard_name, possible_cols in swing_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            if isinstance(value, (int, float)):
                                swing_data[standard_name] = float(value)
                            else:
                                swing_data[standard_name] = str(value)
                            signal[standard_name] = swing_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 9. SUPPORT/RESISTANCE DATA
            sr_data = {}
            sr_fields = {
                'nearest_support': ['nearest_support', 'support_level', 'support_price'],
                'nearest_resistance': ['nearest_resistance', 'resistance_level', 'resistance_price'],
                'distance_to_support_pips': ['distance_to_support_pips', 'support_distance_pips'],
                'distance_to_resistance_pips': ['distance_to_resistance_pips', 'resistance_distance_pips'],
                'support_strength': ['support_strength', 'support_level_strength'],
                'resistance_strength': ['resistance_strength', 'resistance_level_strength'],
                'level_flip_detected': ['level_flip_detected', 'sr_flip_detected'],
                'cluster_risk_level': ['cluster_risk_level', 'sr_cluster_risk']
            }

            for standard_name, possible_cols in sr_fields.items():
                for col in possible_cols:
                    if col in df.columns:
                        try:
                            value = latest[col]
                            if isinstance(value, bool):
                                sr_data[standard_name] = bool(value)
                            elif isinstance(value, (int, float)):
                                sr_data[standard_name] = float(value)
                            else:
                                sr_data[standard_name] = str(value)
                            signal[standard_name] = sr_data[standard_name]
                            break
                        except (ValueError, KeyError):
                            continue

            # 10. ADDITIONAL CONTEXT DATA
            context_fields = [
                'volume_confirmation', 'trend_alignment', 'market_session',
                'consolidation_range_pips', 'bars_since_breakout'
            ]
            for field in context_fields:
                if field in df.columns:
                    try:
                        signal[field] = latest[field]
                    except (ValueError, KeyError):
                        continue

            # 11. CREATE COMPREHENSIVE STRATEGY_INDICATORS JSON
            all_indicators = {}
            all_indicators.update(ema_indicators)
            all_indicators.update(macd_indicators)
            all_indicators.update(kama_indicators)
            all_indicators.update(adx_indicators)
            all_indicators.update(other_indicators)

            # Helper function to clean NaN values
            def clean_nan_values(obj):
                """Recursively replace NaN with None in dict/list structures"""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float):
                    import math
                    return None if math.isnan(obj) or math.isinf(obj) else obj
                else:
                    return obj

            if all_indicators:
                existing_strategy_indicators = signal.get('strategy_indicators', {})
                strategy_name = signal.get('strategy', 'unknown')

                if existing_strategy_indicators:
                    self.logger.info(f"🔍 [{strategy_name}] Found existing strategy_indicators with keys: {list(existing_strategy_indicators.keys())}")
                else:
                    self.logger.info(f"🔍 [{strategy_name}] No existing strategy_indicators found - will use dataframe analysis")

                dataframe_indicators = clean_nan_values({
                    'ema_data': ema_indicators,
                    'macd_data': macd_indicators,
                    'kama_data': kama_indicators,
                    'adx_data': adx_indicators,
                    'swing_data': swing_data,
                    'sr_data': sr_data,
                    'other_indicators': other_indicators,
                    'indicator_count': len(all_indicators),
                    'data_source': 'complete_dataframe_analysis'
                })

                if existing_strategy_indicators:
                    merged_indicators = existing_strategy_indicators.copy()
                    merged_indicators['dataframe_analysis'] = dataframe_indicators
                    signal['strategy_indicators'] = merged_indicators
                    self.logger.info(f"✅ [{strategy_name}] Preserved strategy indicators + added {len(all_indicators)} dataframe indicators")
                else:
                    signal['strategy_indicators'] = dataframe_indicators
                    self.logger.debug(f"📊 Enhanced signal with {len(all_indicators)} indicators + swing/SR data")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error adding complete technical indicators: {e}")
            return signal
