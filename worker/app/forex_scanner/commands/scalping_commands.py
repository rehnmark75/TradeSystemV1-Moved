# commands/scalping_commands.py - Minimal fixed version
import logging

class ScalpingCommands:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def __getattr__(self, name):
        def placeholder(*args, **kwargs):
            self.logger.info(f"📋 ScalpingCommands.{name} called (placeholder)")
            return True
        return placeholder
# commands/scalping_commands.py
"""
Scalping Commands Module
Handles 5-minute scalping operations and debugging
"""

import logging
from typing import List, Dict, Optional

try:
    from core.scanner import ForexScanner
    from core.database import DatabaseManager
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

class ScalpingCommands:
    """Scalping command implementations for 5-minute data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_scalping_scan(self, epic: str = None, mode: str = 'aggressive') -> bool:
        """Run a scan focused on 5-minute scalping signals"""
        self.logger.info(f"🏃 Running 5-minute scalping scan in {mode} mode")
        
        # Temporarily switch to scalping mode
        original_mode = getattr(config, 'ACTIVE_SCALPING_CONFIG', 'aggressive')
        original_enabled = getattr(config, 'SCALPING_STRATEGY_ENABLED', False)
        
        try:
            # Enable scalping for 5m
            config.SCALPING_STRATEGY_ENABLED = True
            config.ACTIVE_SCALPING_CONFIG = mode
            
            # Initialize scanner with 5m scalping-optimized settings
            db_manager = DatabaseManager(config.DATABASE_URL)
            epic_list = [epic] if epic else config.SCALPING_MARKET_CONDITIONS['preferred_pairs']
            
            scanner = ForexScanner(
                db_manager=db_manager,
                epic_list=epic_list,
                scan_interval=60,  # 1 minute scanning for 5m data
                claude_api_key=config.CLAUDE_API_KEY,
                enable_claude_analysis=False,  # Disable for speed
                use_bid_adjustment=True,
                spread_pips=2.0,  # Higher spread tolerance for 5m
                min_confidence=0.65,  # Slightly lower for more 5m signals
                user_timezone=config.USER_TIMEZONE
            )
            
            # Run scan
            signals = scanner.scan_once()
            
            if signals:
                self.logger.info(f"🎯 Found {len(signals)} 5-minute scalping signals:")
                for signal in signals:
                    scalping_mode = signal.get('scalping_mode', 'unknown')
                    entry_reason = signal.get('entry_reason', 'unknown')
                    target_pips = config.get_scalping_config().get('target_pips', 10)
                    stop_pips = config.get_scalping_config().get('stop_loss_pips', 6)
                    
                    self.logger.info(f"  {signal['epic']}: {signal['signal_type']} "
                                   f"({scalping_mode} - {entry_reason})")
                    self.logger.info(f"    Confidence: {signal['confidence_score']:.1%}")
                    self.logger.info(f"    Target: {target_pips} pips, Stop: {stop_pips} pips")
            else:
                self.logger.info("No 5-minute scalping signals detected")
                self._log_scalping_market_state(epic_list[0] if epic_list else 'CS.D.EURUSD.MINI.IP')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 5-minute scalping scan failed: {e}")
            return False
            
        finally:
            # Restore original settings
            config.ACTIVE_SCALPING_CONFIG = original_mode
            config.SCALPING_STRATEGY_ENABLED = original_enabled
    
    def debug_scalping_signal(self, epic: str, mode: str = 'aggressive') -> bool:
        """Debug 5-minute scalping signal detection"""
        self.logger.info(f"🔍 Debugging 5-minute scalping signal for {epic} in {mode} mode")
        
        try:
            # Setup for 5m scalping
            original_mode = getattr(config, 'ACTIVE_SCALPING_CONFIG', 'aggressive')
            original_enabled = getattr(config, 'SCALPING_STRATEGY_ENABLED', False)
            
            config.SCALPING_STRATEGY_ENABLED = True
            config.ACTIVE_SCALPING_CONFIG = mode
            
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get 5-minute scalping signal with higher spread tolerance
            signal = detector.detect_scalping_signals(epic, pair, 2.0, '5m')
            
            # Display debug info
            if signal:
                self._display_scalping_signal_details(signal)
            else:
                self.logger.info("❌ No 5-minute scalping signal detected")
                self._debug_why_no_scalping_signal(detector, epic, pair, mode)
            
            # Restore settings
            config.ACTIVE_SCALPING_CONFIG = original_mode
            config.SCALPING_STRATEGY_ENABLED = original_enabled
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 5-minute scalping debug failed: {e}")
            return False
    
    def _display_scalping_signal_details(self, signal: Dict):
        """Display detailed scalping signal information"""
        self.logger.info("🎯 5-MINUTE SCALPING SIGNAL DETECTED:")
        self.logger.info(f"  Type: {signal['signal_type']}")
        self.logger.info(f"  Mode: {signal.get('scalping_mode', 'unknown')}")
        self.logger.info(f"  Reason: {signal.get('entry_reason', 'unknown')}")
        self.logger.info(f"  Confidence: {signal['confidence_score']:.1%}")
        self.logger.info(f"  Fast EMA: {signal.get('fast_ema', 0):.5f}")
        self.logger.info(f"  Slow EMA: {signal.get('slow_ema', 0):.5f}")
        self.logger.info(f"  Volume OK: {signal.get('volume_confirmed', False)}")
        self.logger.info(f"  Session: {signal.get('current_session', 'unknown')}")
        
        # Risk management info
        config_data = config.get_scalping_config()
        self.logger.info(f"  Target: {config_data['target_pips']} pips")
        self.logger.info(f"  Stop Loss: {config_data['stop_loss_pips']} pips")
        self.logger.info(f"  Risk/Reward: 1:{config_data['target_pips']/config_data['stop_loss_pips']:.1f}")
        
        # Additional context
        if 'atr_5' in signal:
            self.logger.info(f"  5-bar ATR: {signal['atr_5']:.1f} pips")
        if 'recent_range_pips' in signal:
            self.logger.info(f"  Recent Range: {signal['recent_range_pips']:.1f} pips")
        if 'scalping_quality_score' in signal:
            self.logger.info(f"  Quality Score: {signal['scalping_quality_score']:.2f}")
    
    def _debug_why_no_scalping_signal(self, detector, epic: str, pair: str, mode: str):
        """Debug why no scalping signal was detected"""
        try:
            # Get 5m data
            df = detector.data_fetcher.get_enhanced_data(epic, pair, '5m', 48)  # 4 hours of 5m data
            if df is None or len(df) < 2:
                self.logger.error("❌ Insufficient 5-minute data for analysis")
                return
            
            latest = df.iloc[-1]
            config_data = config.get_scalping_config()
            
            # Get EMA values
            fast_ema = latest.get(f'ema_{config_data["fast_ema"]}', 0)
            slow_ema = latest.get(f'ema_{config_data["slow_ema"]}', 0)
            filter_ema = latest.get(f'ema_{config_data["filter_ema"]}', 0) if config_data.get("filter_ema") else None
            
            self.logger.info(f"💡 Current 5-minute market state ({mode} mode):")
            self.logger.info(f"  Price: {latest['close']:.5f}")
            self.logger.info(f"  Fast EMA ({config_data['fast_ema']}): {fast_ema:.5f}")
            self.logger.info(f"  Slow EMA ({config_data['slow_ema']}): {slow_ema:.5f}")
            if filter_ema:
                self.logger.info(f"  Filter EMA ({config_data['filter_ema']}): {filter_ema:.5f}")
            
            # EMA relationships
            self.logger.info(f"  EMA Relationship: {'Fast > Slow' if fast_ema > slow_ema else 'Slow > Fast'}")
            
            # 5m specific market conditions
            candle_range_pips = (latest['high'] - latest['low']) * 10000
            self.logger.info(f"  5m Candle Range: {candle_range_pips:.1f} pips")
            
            # Volume analysis
            volume_ratio = latest.get('volume_ratio_20', 1.0)
            self.logger.info(f"  Volume Ratio: {volume_ratio:.2f}x")
            if volume_ratio < 1.2:
                self.logger.info("  ⚠️ Low volume - may affect scalping signal quality")
            
            # Spread check
            max_spread = config_data.get('max_spread_pips', 2.0)
            self.logger.info(f"  Max allowed spread: {max_spread} pips")
            
            # Session check
            session_ok = config.is_scalping_session()
            self.logger.info(f"  Good scalping session: {'✅' if session_ok else '❌'}")
            
            # Market quality hints
            if candle_range_pips < 2:
                self.logger.info("  💡 Very tight range - market may be too quiet for scalping")
            elif candle_range_pips > 20:
                self.logger.info("  💡 Very wide range - market may be too volatile for scalping")
            
        except Exception as e:
            self.logger.error(f"❌ Debug analysis failed: {e}")
    
    def _log_scalping_market_state(self, epic: str):
        """Log current market state for scalping"""
        try:
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            df = detector.data_fetcher.get_enhanced_data(epic, pair, '5m', 12)  # Last hour
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]
                
                # Calculate recent volatility
                recent_ranges = [(row['high'] - row['low']) * 10000 for _, row in df.tail(5).iterrows()]
                avg_range = sum(recent_ranges) / len(recent_ranges)
                
                self.logger.info(f"💡 Current 5m market conditions for {epic}:")
                self.logger.info(f"   Average range (5 bars): {avg_range:.1f} pips")
                self.logger.info(f"   Volume ratio: {latest.get('volume_ratio_20', 1.0):.2f}x")
                self.logger.info(f"   Current session: {detector._get_current_session() if hasattr(detector, '_get_current_session') else 'unknown'}")
                
                if avg_range < 3:
                    self.logger.info("   📊 Market is very quiet - consider waiting for volatility")
                elif avg_range > 15:
                    self.logger.info("   📊 Market is highly volatile - use conservative mode")
                else:
                    self.logger.info("   📊 Market conditions look suitable for scalping")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log market state: {e}")
    
    def run_scalping_backtest(self, epic: str = None, days: int = 1, mode: str = 'aggressive') -> bool:
        """Run scalping-focused backtest"""
        self.logger.info(f"📊 Running 5-minute scalping backtest ({mode} mode)")
        
        try:
            # Import here to avoid circular imports
            from commands.backtest_commands import BacktestCommands
            
            # Temporarily enable scalping
            original_mode = getattr(config, 'ACTIVE_SCALPING_CONFIG', 'aggressive')
            original_enabled = getattr(config, 'SCALPING_STRATEGY_ENABLED', False)
            
            config.SCALPING_STRATEGY_ENABLED = True
            config.ACTIVE_SCALPING_CONFIG = mode
            
            # Run backtest with scalping settings
            backtest_commands = BacktestCommands()
            success = backtest_commands.run_backtest(
                epic=epic, 
                days=days, 
                timeframe='5m',
                show_signals=True,
                ema_config=None  # Use scalping-specific EMAs
            )
            
            # Restore settings
            config.ACTIVE_SCALPING_CONFIG = original_mode
            config.SCALPING_STRATEGY_ENABLED = original_enabled
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Scalping backtest failed: {e}")
            return False
    
    def get_scalping_config_info(self) -> Dict:
        """Get current scalping configuration information"""
        try:
            active_mode = getattr(config, 'ACTIVE_SCALPING_CONFIG', 'aggressive')
            enabled = getattr(config, 'SCALPING_STRATEGY_ENABLED', False)
            
            if not enabled:
                return {'enabled': False, 'message': 'Scalping strategy is disabled'}
            
            scalping_config = config.get_scalping_config()
            
            return {
                'enabled': True,
                'active_mode': active_mode,
                'config': scalping_config,
                'timeframe': config.SCALPING_TIMEFRAME,
                'preferred_pairs': config.SCALPING_MARKET_CONDITIONS['preferred_pairs'],
                'risk_management': config.SCALPING_RISK_MANAGEMENT
            }
            
        except Exception as e:
            return {'error': f"Failed to get scalping config: {e}"}
    
    def test_scalping_setup(self) -> bool:
        """Test scalping configuration and setup"""
        self.logger.info("🧪 Testing 5-minute scalping setup...")
        
        try:
            # Test scalping configuration
            config_info = self.get_scalping_config_info()
            if 'error' in config_info:
                self.logger.error(f"❌ Scalping config error: {config_info['error']}")
                return False
            
            if not config_info['enabled']:
                self.logger.warning("⚠️ Scalping strategy is disabled")
                return False
            
            self.logger.info("✅ Scalping configuration OK")
            self.logger.info(f"   Active mode: {config_info['active_mode']}")
            self.logger.info(f"   Target pips: {config_info['config']['target_pips']}")
            self.logger.info(f"   Stop pips: {config_info['config']['stop_loss_pips']}")
            
            # Test scalping strategy initialization
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            if not hasattr(detector, 'scalping_strategy') or detector.scalping_strategy is None:
                self.logger.error("❌ Scalping strategy not initialized in SignalDetector")
                return False
            
            self.logger.info("✅ Scalping strategy initialized OK")
            
            # Test data availability
            test_epic = config_info['preferred_pairs'][0]
            pair_info = config.PAIR_INFO.get(test_epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            df = detector.data_fetcher.get_enhanced_data(test_epic, pair, '5m', 24)
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient 5-minute data for scalping")
                return False
            
            self.logger.info(f"✅ 5-minute data available: {len(df)} bars")
            
            # Test session detection
            session_ok = config.is_scalping_session()
            self.logger.info(f"✅ Session check: {'Good for scalping' if session_ok else 'Outside preferred hours'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Scalping setup test failed: {e}")
            return False