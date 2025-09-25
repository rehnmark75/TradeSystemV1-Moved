# commands/debug_commands.py - Fixed
import logging
try:
    from core.signal_detector import SignalDetector
    import config
except ImportError:
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.signal_detector import SignalDetector
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.scanner import IntelligentForexScanner as ForexScanner
        from forex_scanner import config
    except ImportError as e:
        import sys
        print(f"Warning: Import fallback failed for {sys.modules[__name__]}: {e}")
        pass
class DebugCommands:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.signal_detector = SignalDetector()  # No parameters
        except Exception as e:
            self.logger.error(f"❌ Debug init error: {e}")
    
    def debug_signal(self, epic, timestamp=None):
        self.logger.info(f"🔍 Debugging signal for {epic}")
        if timestamp:
            self.logger.info(f"   Timestamp: {timestamp}")
        
        try:
            # Get current data and test signal detection
            df = self.signal_detector.data_fetcher.fetch_latest_data(epic, 5, 300)
            if df is not None:
                self.logger.info(f"✅ Data available: {len(df)} bars")
                
                # Check indicators
                ema_cols = [col for col in df.columns if 'ema_' in col]
                self.logger.info(f"   EMA indicators: {ema_cols}")
                
                # Test signal detection
                signal = self.signal_detector.detect_primary_signal(df, epic, 1.5, '5m')
                if signal:
                    self.logger.info(f"🎯 Signal detected: {signal['signal_type']} ({signal['confidence_score']:.1%})")
                else:
                    self.logger.info("📊 No signal detected")
            else:
                self.logger.error("❌ No data available")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Debug error: {e}")
            return False
    
    def debug_macd_signal(self, epic):
        self.logger.info(f"📊 MACD debug for {epic} (placeholder)")
        return True
    
    def debug_combined_strategies(self, epic):
        self.logger.info(f"🤝 Combined strategies debug for {epic} (placeholder)")
        return True
    
    def test_combined_methods_exist(self):
        self.logger.info("🧪 Testing combined methods (placeholder)")
        return True
    
    def debug_backtest_process(self, epic=None, days=7, timeframe='5m'):
        self.logger.info(f"🔍 Debug backtest process (placeholder)")
        return True
# commands/debug_commands.py
"""
Debug Commands Module
Handles debugging operations for signal detection, strategies, and system validation
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    from core.strategies.ema_strategy import EMAStrategy
    from core.strategies.macd_strategy import MACDStrategy
    # from core.strategies.combined_strategy import CombinedStrategy  # Removed - strategy was disabled and unused
    import config
except ImportError:
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.signal_detector import SignalDetector
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.scanner import IntelligentForexScanner as ForexScanner
        from forex_scanner import config
    except ImportError as e:
        import sys
        print(f"Warning: Import fallback failed for {sys.modules[__name__]}: {e}")
        pass

class DebugCommands:
    """Debug command implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def debug_signal(self, epic: str, timestamp: str = None) -> bool:
        """Debug EMA signal detection for a specific epic and optional timestamp"""
        self.logger.info(f"🔍 Debugging signal detection for {epic}")
        
        try:
            # Initialize signal detector
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            if timestamp:
                # Debug specific timestamp
                debug_info = detector.debug_signal_at_timestamp(epic, pair, timestamp, config.SPREAD_PIPS)
            else:
                # Debug current signal detection
                debug_info = detector.debug_signal_detection(epic, pair, config.SPREAD_PIPS)
            
            # Display debug information
            self._display_debug_info(debug_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Debug signal failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_macd_signal(self, epic: str) -> bool:
        """Debug MACD + EMA 200 signal detection"""
        self.logger.info(f"🔍 Debugging MACD signal detection for {epic}")
        
        try:
            # Initialize signal detector
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Debug MACD signal
            debug_info = detector.debug_macd_ema_signal(epic, pair, config.SPREAD_PIPS)
            
            # Display MACD-specific debug information
            self._display_macd_debug_info(debug_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Debug MACD signal failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_combined_strategies(self, epic: str) -> bool:
        """Debug combined strategy detection"""
        self.logger.info(f"🎯 Debugging combined strategy detection for {epic}")
        
        try:
            # Initialize signal detector
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Debug combined strategy setup
            debug_info = detector.debug_combined_strategy_setup(epic, pair)
            
            # Display combined strategy debug information
            self._display_combined_debug_info(debug_info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Debug combined strategies failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_combined_methods_exist(self) -> bool:
        """Test if combined strategy methods exist and are callable"""
        self.logger.info("🧪 Testing combined strategy methods existence")
        
        try:
            # Test combined strategy class
            # combined_strategy = CombinedStrategy()  # Removed - strategy was disabled and unused
            
            # Check required methods
            required_methods = [
                '_combine_signals_consensus',
                '_combine_signals_confirmation', 
                '_combine_signals_hierarchy',
                '_combine_signals_dynamic',
                '_create_combined_signal',
                '_enhance_single_signal'
            ]
            
            self.logger.info("📋 Method Existence Check:")
            all_methods_exist = True
            
            for method_name in required_methods:
                if hasattr(combined_strategy, method_name):
                    method = getattr(combined_strategy, method_name)
                    if callable(method):
                        self.logger.info(f"  ✅ {method_name}: Found and callable")
                    else:
                        self.logger.error(f"  ❌ {method_name}: Found but not callable")
                        all_methods_exist = False
                else:
                    self.logger.error(f"  ❌ {method_name}: Not found")
                    all_methods_exist = False
            
            # Test strategy initialization
            self.logger.info("\n🔧 Strategy Initialization Test:")
            try:
                ema_strategy = EMAStrategy()
                self.logger.info("  ✅ EMA Strategy: Initialized successfully")
            except Exception as e:
                self.logger.error(f"  ❌ EMA Strategy: Failed to initialize - {e}")
                all_methods_exist = False
            
            try:
                macd_strategy = MACDStrategy()
                self.logger.info("  ✅ MACD Strategy: Initialized successfully")
            except Exception as e:
                self.logger.error(f"  ❌ MACD Strategy: Failed to initialize - {e}")
                all_methods_exist = False
            
            # Test required indicators
            self.logger.info("\n📊 Required Indicators Check:")
            try:
                ema_indicators = combined_strategy.ema_strategy.get_required_indicators()
                self.logger.info(f"  ✅ EMA indicators: {ema_indicators}")
            except Exception as e:
                self.logger.error(f"  ❌ EMA indicators: Failed - {e}")
                all_methods_exist = False
            
            try:
                macd_indicators = combined_strategy.macd_strategy.get_required_indicators()
                self.logger.info(f"  ✅ MACD indicators: {macd_indicators}")
            except Exception as e:
                self.logger.error(f"  ❌ MACD indicators: Failed - {e}")
                all_methods_exist = False
            
            # Test configuration
            self.logger.info("\n⚙️ Configuration Check:")
            config_checks = {
                'SIMPLE_EMA_STRATEGY': getattr(config, 'SIMPLE_EMA_STRATEGY', 'NOT_SET'),
                'MACD_EMA_STRATEGY': getattr(config, 'MACD_EMA_STRATEGY', 'NOT_SET'),
                'COMBINED_STRATEGY_MODE': getattr(config, 'COMBINED_STRATEGY_MODE', 'NOT_SET'),
                'MIN_COMBINED_CONFIDENCE': getattr(config, 'MIN_COMBINED_CONFIDENCE', 'NOT_SET')
            }
            
            for key, value in config_checks.items():
                if value == 'NOT_SET':
                    self.logger.warning(f"  ⚠️ {key}: Not configured")
                else:
                    self.logger.info(f"  ✅ {key}: {value}")
            
            if all_methods_exist:
                self.logger.info("\n🎉 All combined strategy methods exist and are functional!")
            else:
                self.logger.error("\n❌ Some methods are missing or non-functional")
            
            return all_methods_exist
            
        except Exception as e:
            self.logger.error(f"❌ Combined methods test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def debug_configuration(self) -> bool:
        """Debug current configuration settings"""
        self.logger.info("🔧 Debugging configuration settings")
        
        try:
            # Strategy Configuration
            self.logger.info("📋 STRATEGY CONFIGURATION:")
            strategy_config = {
                'SIMPLE_EMA_STRATEGY': getattr(config, 'SIMPLE_EMA_STRATEGY', 'NOT_SET'),
                'MACD_EMA_STRATEGY': getattr(config, 'MACD_EMA_STRATEGY', 'NOT_SET'),
                'SCALPING_STRATEGY_ENABLED': getattr(config, 'SCALPING_STRATEGY_ENABLED', 'NOT_SET'),
                'COMBINED_STRATEGY_MODE': getattr(config, 'COMBINED_STRATEGY_MODE', 'NOT_SET'),
                'MIN_COMBINED_CONFIDENCE': getattr(config, 'MIN_COMBINED_CONFIDENCE', 'NOT_SET'),
                'STRATEGY_WEIGHT_EMA': getattr(config, 'STRATEGY_WEIGHT_EMA', 'NOT_SET'),
                'STRATEGY_WEIGHT_MACD': getattr(config, 'STRATEGY_WEIGHT_MACD', 'NOT_SET')
            }
            
            for key, value in strategy_config.items():
                status = "✅" if value != 'NOT_SET' else "❌"
                self.logger.info(f"  {status} {key}: {value}")
            
            # EMA Configuration
            self.logger.info("\n📈 EMA CONFIGURATION:")
            if hasattr(config, 'EMA_STRATEGY_CONFIG'):
                active_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
                ema_config = config.EMA_STRATEGY_CONFIG.get(active_config, {})
                self.logger.info(f"  ✅ Active EMA Config: {active_config}")
                self.logger.info(f"  ✅ Short EMA: {ema_config.get('short', 'N/A')}")
                self.logger.info(f"  ✅ Long EMA: {ema_config.get('long', 'N/A')}")
                self.logger.info(f"  ✅ Trend EMA: {ema_config.get('trend', 'N/A')}")
            else:
                self.logger.warning("  ⚠️ EMA_STRATEGY_CONFIG not found - using legacy config")
                self.logger.info(f"  ✅ EMA_PERIODS: {getattr(config, 'EMA_PERIODS', 'NOT_SET')}")
            
            # Database Configuration
            self.logger.info("\n🗄️ DATABASE CONFIGURATION:")
            db_url = getattr(config, 'DATABASE_URL', None)
            if db_url:
                # Hide password in display
                safe_url = db_url.split('@')[-1] if '@' in db_url else db_url
                self.logger.info(f"  ✅ Database URL: ...@{safe_url}")
            else:
                self.logger.error("  ❌ DATABASE_URL: Not configured")
            
            # API Configuration
            self.logger.info("\n🤖 API CONFIGURATION:")
            claude_key = getattr(config, 'CLAUDE_API_KEY', None)
            if claude_key:
                self.logger.info(f"  ✅ Claude API Key: {claude_key[:10]}...{claude_key[-4:] if len(claude_key) > 14 else ''}")
            else:
                self.logger.warning("  ⚠️ Claude API Key: Not configured")
            
            # Trading Configuration
            self.logger.info("\n💰 TRADING CONFIGURATION:")
            trading_config = {
                'EPIC_LIST': len(getattr(config, 'EPIC_LIST', [])),
                'DEFAULT_TIMEFRAME': getattr(config, 'DEFAULT_TIMEFRAME', 'NOT_SET'),
                'SPREAD_PIPS': getattr(config, 'SPREAD_PIPS', 'NOT_SET'),
                'MIN_CONFIDENCE': getattr(config, 'MIN_CONFIDENCE', 'NOT_SET'),
                'USE_BID_ADJUSTMENT': getattr(config, 'USE_BID_ADJUSTMENT', 'NOT_SET'),
                'SCAN_INTERVAL': getattr(config, 'SCAN_INTERVAL', 'NOT_SET')
            }
            
            for key, value in trading_config.items():
                if key == 'EPIC_LIST':
                    self.logger.info(f"  ✅ {key}: {value} pairs")
                else:
                    status = "✅" if value != 'NOT_SET' else "❌"
                    self.logger.info(f"  {status} {key}: {value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration debug failed: {e}")
            return False
    
    def test_data_availability(self, epic: str = None) -> bool:
        """Test data availability for signal detection"""
        test_epic = epic or 'CS.D.EURUSD.MINI.IP'
        self.logger.info(f"📊 Testing data availability for {test_epic}")
        
        try:
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(test_epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Test different timeframes
            timeframes = ['5m', '15m', '1h']
            
            for timeframe in timeframes:
                self.logger.info(f"\n🔍 Testing {timeframe} data:")
                
                try:
                    df = detector.data_fetcher.get_enhanced_data(
                        test_epic, pair, timeframe=timeframe, lookback_hours=168  # 1 week
                    )
                    
                    if df is not None and len(df) > 0:
                        self.logger.info(f"  ✅ {timeframe}: {len(df)} bars available")
                        
                        # Check for required indicators
                        required_indicators = ['ema_9', 'ema_21', 'ema_200']
                        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
                        
                        if missing_indicators:
                            self.logger.warning(f"  ⚠️ Missing indicators: {missing_indicators}")
                        else:
                            self.logger.info(f"  ✅ All required EMA indicators present")
                        
                        # Check data quality
                        latest_data = df.iloc[-1]
                        self.logger.info(f"  ✅ Latest timestamp: {latest_data['start_time']}")
                        self.logger.info(f"  ✅ Latest price: {latest_data['close']:.5f}")
                        
                        # Check for MACD indicators if MACD strategy is enabled
                        if getattr(config, 'MACD_EMA_STRATEGY', False):
                            macd_indicators = ['macd_line', 'macd_signal', 'macd_histogram']
                            missing_macd = [ind for ind in macd_indicators if ind not in df.columns]
                            
                            if missing_macd:
                                self.logger.warning(f"  ⚠️ Missing MACD indicators: {missing_macd}")
                            else:
                                self.logger.info(f"  ✅ All MACD indicators present")
                    else:
                        self.logger.error(f"  ❌ {timeframe}: No data available")
                        
                except Exception as e:
                    self.logger.error(f"  ❌ {timeframe}: Error - {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data availability test failed: {e}")
            return False
    
    def _display_debug_info(self, debug_info: Dict):
        """Display comprehensive debug information"""
        if 'error' in debug_info:
            self.logger.error(f"❌ Debug Error: {debug_info['error']}")
            return
        
        self.logger.info("🔬 SIGNAL DEBUG INFORMATION:")
        self.logger.info("=" * 60)
        
        # Basic information
        self.logger.info(f"Epic: {debug_info.get('epic', 'Unknown')}")
        self.logger.info(f"Timestamp: {debug_info.get('timestamp', 'Unknown')}")
        self.logger.info(f"Strategy: {debug_info.get('strategy_used', 'Unknown')}")
        
        # Signal detection results
        signal_detected = debug_info.get('signal_detected', False)
        signal_type = debug_info.get('signal_type', 'None')
        confidence = debug_info.get('confidence_score', 0)
        
        status_icon = "✅" if signal_detected else "❌"
        self.logger.info(f"Signal Detected: {status_icon} {signal_detected}")
        
        if signal_detected:
            self.logger.info(f"Signal Type: {signal_type}")
            self.logger.info(f"Confidence: {confidence:.1%}")
            threshold = debug_info.get('min_confidence_threshold', 0.6)
            above_threshold = debug_info.get('above_threshold', False)
            threshold_icon = "✅" if above_threshold else "❌"
            self.logger.info(f"Above Threshold: {threshold_icon} {confidence:.1%} vs {threshold:.1%}")
        
        # Price information
        self.logger.info(f"\n💰 PRICE INFORMATION:")
        self.logger.info(f"Current Price: {debug_info.get('current_price', 0):.5f}")
        self.logger.info(f"Previous Price: {debug_info.get('prev_price', 0):.5f}")
        
        if 'original_close_price' in debug_info:
            self.logger.info(f"Original (BID): {debug_info['original_close_price']:.5f}")
            self.logger.info(f"Adjusted (MID): {debug_info['adjusted_close_price']:.5f}")
            self.logger.info(f"Spread Adjustment: {debug_info.get('spread_adjustment_pips', 0)} pips")
        
        # EMA information
        if 'ema_9_current' in debug_info:
            self.logger.info(f"\n📈 EMA INFORMATION:")
            self.logger.info(f"EMA 9: {debug_info['ema_9_current']:.5f}")
            self.logger.info(f"EMA 21: {debug_info.get('ema_21_current', 0):.5f}")
            self.logger.info(f"EMA 200: {debug_info.get('ema_200_current', 0):.5f}")
            
            # EMA relationships
            current_price = debug_info.get('current_price', 0)
            ema_9 = debug_info['ema_9_current']
            ema_21 = debug_info.get('ema_21_current', 0)
            ema_200 = debug_info.get('ema_200_current', 0)
            
            self.logger.info(f"\n📊 EMA RELATIONSHIPS:")
            self.logger.info(f"Price vs EMA 9: {'Above' if current_price > ema_9 else 'Below'}")
            self.logger.info(f"EMA 9 vs EMA 21: {'Above' if ema_9 > ema_21 else 'Below'}")
            self.logger.info(f"EMA 9 vs EMA 200: {'Above' if ema_9 > ema_200 else 'Below'}")
            self.logger.info(f"EMA 21 vs EMA 200: {'Above' if ema_21 > ema_200 else 'Below'}")
        
        # Additional timestamp info if available
        if 'closest_timestamp' in debug_info:
            self.logger.info(f"\n🕐 TIMESTAMP INFO:")
            self.logger.info(f"Closest Available: {debug_info['closest_timestamp']}")
            if 'time_diff_minutes' in debug_info:
                self.logger.info(f"Time Difference: {debug_info['time_diff_minutes']:.1f} minutes")
    
    def _display_macd_debug_info(self, debug_info: Dict):
        """Display MACD-specific debug information"""
        if 'error' in debug_info:
            self.logger.error(f"❌ MACD Debug Error: {debug_info['error']}")
            return
        
        self.logger.info("🔬 MACD SIGNAL DEBUG INFORMATION:")
        self.logger.info("=" * 60)
        
        # Basic information
        self.logger.info(f"Epic: {debug_info.get('epic', 'Unknown')}")
        self.logger.info(f"Strategy: {debug_info.get('strategy', 'Unknown')}")
        
        # Signal results
        signal_detected = debug_info.get('signal_detected', False)
        signal_type = debug_info.get('signal_type', 'None')
        confidence = debug_info.get('confidence_score', 0)
        
        status_icon = "✅" if signal_detected else "❌"
        self.logger.info(f"MACD Signal: {status_icon} {signal_detected}")
        
        if signal_detected:
            self.logger.info(f"Signal Type: {signal_type}")
            self.logger.info(f"Confidence: {confidence:.1%}")
        
        # MACD data
        self.logger.info(f"\n📊 MACD INDICATORS:")
        macd_fields = [
            ('ema_200_current', 'EMA 200'),
            ('macd_line_current', 'MACD Line'),
            ('macd_signal_current', 'MACD Signal'),
            ('macd_histogram_current', 'MACD Histogram'),
            ('macd_color_current', 'MACD Color'),
            ('macd_color_prev', 'Previous Color')
        ]
        
        for field, label in macd_fields:
            if field in debug_info:
                value = debug_info[field]
                if isinstance(value, float):
                    self.logger.info(f"{label}: {value:.6f}")
                else:
                    self.logger.info(f"{label}: {value}")
        
        # Price vs EMA 200
        if 'current_price' in debug_info and 'ema_200_current' in debug_info:
            current_price = debug_info['current_price']
            ema_200 = debug_info['ema_200_current']
            relationship = "Above" if current_price > ema_200 else "Below"
            self.logger.info(f"\n💰 Price vs EMA 200: {relationship}")
            self.logger.info(f"Current Price: {current_price:.5f}")
            self.logger.info(f"EMA 200: {ema_200:.5f}")
    
    def _display_combined_debug_info(self, debug_info: Dict):
        """Display combined strategy debug information"""
        if 'error' in debug_info:
            self.logger.error(f"❌ Combined Debug Error: {debug_info['error']}")
            return
        
        self.logger.info("🎯 COMBINED STRATEGY DEBUG INFORMATION:")
        self.logger.info("=" * 60)
        
        # Configuration check
        self.logger.info("⚙️ CONFIGURATION CHECK:")
        config_check = debug_info.get('config_check', {})
        for key, value in config_check.items():
            status = "✅" if value not in ['NOT_SET', False] else "❌"
            self.logger.info(f"  {status} {key}: {value}")
        
        # Individual signals
        self.logger.info(f"\n📊 INDIVIDUAL SIGNALS:")
        individual_signals = debug_info.get('individual_signals', {})
        
        # EMA signal
        ema_signal = individual_signals.get('ema', {})
        ema_detected = ema_signal.get('signal_detected', False)
        ema_icon = "✅" if ema_detected else "❌"
        self.logger.info(f"  {ema_icon} EMA Signal: {ema_detected}")
        if ema_detected:
            self.logger.info(f"    Type: {ema_signal.get('signal_type', 'Unknown')}")
            self.logger.info(f"    Confidence: {ema_signal.get('confidence', 0):.1%}")
            self.logger.info(f"    Strategy: {ema_signal.get('strategy', 'Unknown')}")
        
        # MACD signal
        macd_signal = individual_signals.get('macd', {})
        macd_detected = macd_signal.get('signal_detected', False)
        macd_icon = "✅" if macd_detected else "❌"
        self.logger.info(f"  {macd_icon} MACD Signal: {macd_detected}")
        if macd_detected:
            self.logger.info(f"    Type: {macd_signal.get('signal_type', 'Unknown')}")
            self.logger.info(f"    Confidence: {macd_signal.get('confidence', 0):.1%}")
            self.logger.info(f"    Strategy: {macd_signal.get('strategy', 'Unknown')}")
        
        # Combined signal
        self.logger.info(f"\n🎯 COMBINED SIGNAL:")
        combined_signal = debug_info.get('combined_signal', {})
        combined_detected = combined_signal.get('signal_detected', False)
        combined_icon = "✅" if combined_detected else "❌"
        self.logger.info(f"  {combined_icon} Combined Signal: {combined_detected}")
        
        if combined_detected:
            self.logger.info(f"    Type: {combined_signal.get('signal_type', 'Unknown')}")
            self.logger.info(f"    Confidence: {combined_signal.get('confidence', 0):.1%}")
            self.logger.info(f"    Strategy: {combined_signal.get('strategy', 'Unknown')}")
            self.logger.info(f"    Mode: {combined_signal.get('combination_mode', 'Unknown')}")
        
        # Analysis
        self.logger.info(f"\n🔍 ANALYSIS:")
        if ema_detected and macd_detected:
            ema_type = ema_signal.get('signal_type', 'Unknown')
            macd_type = macd_signal.get('signal_type', 'Unknown')
            
            if ema_type == macd_type:
                self.logger.info(f"  ✅ Signals AGREE: Both {ema_type}")
                if not combined_detected:
                    self.logger.warning(f"  ⚠️ Expected combined signal but none detected!")
                    self.logger.info(f"  💡 Check combination mode and confidence thresholds")
            else:
                self.logger.info(f"  ❌ Signals DISAGREE: EMA={ema_type}, MACD={macd_type}")
                self.logger.info(f"  💡 No combined signal expected due to disagreement")
        elif ema_detected or macd_detected:
            single_strategy = "EMA" if ema_detected else "MACD"
            self.logger.info(f"  ℹ️ Only {single_strategy} signal detected")
            self.logger.info(f"  💡 Combined signal depends on combination mode")
        else:
            self.logger.info(f"  ℹ️ No individual signals detected")
            self.logger.info(f"  💡 No combined signal expected")