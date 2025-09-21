# main.py
"""
Forex Scanner Main Entry Point
Updated to include Bollinger Bands + Supertrend strategy testing and KAMA strategy support
NOW WITH SMART MONEY COMMANDS (Phase 1 & 2)
"""

import argparse
import logging
from typing import Optional
import sys
try:
    import config
except ImportError:
    from forex_scanner import config

# Command modules
try:
    from commands.scanner_commands import ScannerCommands
except ImportError:
    from forex_scanner.commands.scanner_commands import ScannerCommands
try:
    from commands.debug_commands import DebugCommands
    from commands.backtest_commands import BacktestCommands
    from commands.scalping_commands import ScalpingCommands
    from commands.claude_commands import ClaudeCommands
    from commands.analysis_commands import AnalysisCommands
except ImportError:
    from forex_scanner.commands.debug_commands import DebugCommands
    from forex_scanner.commands.backtest_commands import BacktestCommands
    from forex_scanner.commands.scalping_commands import ScalpingCommands
    from forex_scanner.commands.claude_commands import ClaudeCommands
    from forex_scanner.commands.analysis_commands import AnalysisCommands

# Check if KAMA commands are available
try:
    try:
        from commands.kama_commands import KAMACommands
    except ImportError:
        from forex_scanner.commands.kama_commands import KAMACommands
    KAMA_COMMANDS_AVAILABLE = True
except ImportError:
    KAMA_COMMANDS_AVAILABLE = False

# Check if dynamic config commands are available
try:
    try:
        from commands.dynamic_config_commands import DynamicConfigCommands
    except ImportError:
        from forex_scanner.commands.dynamic_config_commands import DynamicConfigCommands
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIG_AVAILABLE = False

# NEW: Check if smart money commands are available
try:
    try:
        from commands.smart_money_commands import SmartMoneyCommands
    except ImportError:
        from forex_scanner.commands.smart_money_commands import SmartMoneyCommands
    SMART_MONEY_COMMANDS_AVAILABLE = True
except ImportError:
    SMART_MONEY_COMMANDS_AVAILABLE = False


class ForexScannerCLI:
    """Main CLI interface for Forex Scanner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize command modules
        self.scanner_commands = ScannerCommands()
        self.debug_commands = DebugCommands()
        self.backtest_commands = BacktestCommands()
        self.scalping_commands = ScalpingCommands()
        self.claude_commands = ClaudeCommands()
        self.analysis_commands = AnalysisCommands()
        
        # Initialize optional command modules
        if KAMA_COMMANDS_AVAILABLE:
            self.kama_commands = KAMACommands()
        else:
            self.kama_commands = None
            
        if DYNAMIC_CONFIG_AVAILABLE:
            self.dynamic_config_commands = DynamicConfigCommands()
        else:
            self.dynamic_config_commands = None
            
        # NEW: Initialize smart money command modules
        if SMART_MONEY_COMMANDS_AVAILABLE:
            self.smart_money_commands = SmartMoneyCommands()
        else:
            self.smart_money_commands = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands"""
        parser = argparse.ArgumentParser(
            description='Forex Scanner - Technical Analysis & Signal Detection with Smart Money Concepts',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py scan                                    # Single scan
  python main.py live                                    # Live scanning
  python main.py backtest --epic CS.D.EURUSD.CEEM.IP    # Backtest
  python main.py debug --epic CS.D.EURUSD.CEEM.IP       # Debug signals
  python main.py debug-bb-supertrend --epic CS.D.EURUSD.CEEM.IP  # Debug BB+Supertrend
  python main.py test-bb-supertrend --epic CS.D.EURUSD.CEEM.IP   # Test BB+Supertrend
  python main.py backtest-kama --epic CS.D.EURUSD.CEEM.IP --days 14  # KAMA Backtest
  
  # NEW: Smart Money Commands
  python main.py debug-smart-money --epic CS.D.EURUSD.CEEM.IP    # Debug smart money analysis
  python main.py compare-strategies --epic CS.D.EURUSD.CEEM.IP --days 7  # Compare regular vs smart
  python main.py test-smart-validation --epic CS.D.EURUSD.CEEM.IP --signal-type BUY --price 1.0950
  python main.py smart-money-status                              # Check smart money system status
            """
        )
        
        # Global options
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
        parser.add_argument('--config-check', action='store_true', help='Show config and exit')
        
        # Command
        parser.add_argument('command', choices=self._get_available_commands(),
                          help='Command to execute')
        
        # Epic and timeframe options
        parser.add_argument('--epic', type=str, help='Instrument epic to analyze')
        parser.add_argument('--timeframe', type=str, default='15m', 
                          choices=['5m', '15m', '1h'], help='Timeframe for analysis')
        
        # Time-based options
        parser.add_argument('--timestamp', type=str, 
                          help='Specific timestamp (format: "YYYY-MM-DD HH:MM")')
        parser.add_argument('--days', type=int, help='Number of days to backtest')
        
        # Strategy options
        parser.add_argument('--ema-config', type=int, help='EMA config to use')
        parser.add_argument('--ema-configs', type=int, nargs='+', 
                          help='List of EMA configs to compare')
        parser.add_argument('--scalping-mode', type=str, 
                          choices=['ultra_fast', 'aggressive', 'conservative', 'dual_ma'],
                          help='Scalping mode to use')
        
        # Analysis options
        parser.add_argument('--show-signals', action='store_true', 
                          help='Show detailed signal list')
        parser.add_argument('--bb-analysis', action='store_true', 
                          help='Include Bollinger Band analysis')
        parser.add_argument('--no-future', action='store_true', 
                          help='Skip future data in analysis')
        parser.add_argument('--max-analyses', type=int, default=5,
                          help='Max batch size for Claude commands')
        
        # BB+Supertrend specific options
        parser.add_argument('--bb-period', type=int, default=20,
                          help='Bollinger Bands period (default: 20)')
        parser.add_argument('--bb-std-dev', type=float, default=2.0,
                          help='Bollinger Bands standard deviation (default: 2.0)')
        parser.add_argument('--supertrend-period', type=int, default=10,
                          help='Supertrend period (default: 10)')
        parser.add_argument('--supertrend-multiplier', type=float, default=3.0,
                          help='Supertrend multiplier (default: 3.0)')
        
        # KAMA specific options
        parser.add_argument('--kama-config', type=str, default='default',
                          choices=['default', 'conservative', 'aggressive', 'testing'],
                          help='KAMA configuration to use (default: default)')
        parser.add_argument('--kama-min-efficiency', type=float,
                          help='Override KAMA minimum efficiency ratio')
        parser.add_argument('--kama-trend-threshold', type=float,
                          help='Override KAMA trend threshold')
        
        # NEW: Smart Money specific options
        parser.add_argument('--signal-type', type=str, 
                          choices=['BUY', 'SELL', 'BULL', 'BEAR'],
                          help='Signal type for smart money validation testing')
        parser.add_argument('--price', type=float,
                          help='Price level for smart money validation testing')
        parser.add_argument('--enable-structure', action='store_true', default=True,
                          help='Enable market structure analysis (default: True)')
        parser.add_argument('--disable-structure', action='store_true',
                          help='Disable market structure analysis')
        parser.add_argument('--enable-order-flow', action='store_true', default=True,
                          help='Enable order flow analysis (default: True)')
        parser.add_argument('--disable-order-flow', action='store_true',
                          help='Disable order flow analysis')
        parser.add_argument('--strategies', type=str, nargs='+',
                          choices=['ema', 'smart_money_ema', 'macd', 'smart_money_macd'],
                          help='Specific strategies to compare')
        
        return parser
    
    def _get_available_commands(self) -> list:
        """Get list of available commands based on installed modules"""
        base_commands = [
            # Scanner commands
            'scan', 'live',
            
            # Backtest commands
            'backtest', 'compare-ema-configs', 'compare-strategies',   
            
            # Debug commands
            'debug', 'debug-macd', 'debug-combined', 'test-methods',
            
            # BB+Supertrend commands
            'debug-bb-supertrend', 'test-bb-supertrend', 'backtest-bb-supertrend',
            
            # Scalping commands
            'scalp', 'debug-scalping',
            
            # Claude commands
            'test-claude', 'claude-timestamp', 'claude-batch',
            
            # Analysis commands
            'test-bb', 'compare-bb', 'list-ema-configs'
        ]
        
        # Add KAMA commands if available
        if KAMA_COMMANDS_AVAILABLE:
            base_commands.extend([
                'test-kama', 'debug-kama', 'backtest-kama', 'kama-signals',
                'kama-compare', 'kama-optimization'
            ])
        
        # Add dynamic config commands if available
        if DYNAMIC_CONFIG_AVAILABLE:
            base_commands.extend([
                'show-configs', 'config-performance', 'optimize-configs',
                'market-analysis', 'test-config-selection', 'config-settings'
            ])
        
        # NEW: Add smart money commands if available
        if SMART_MONEY_COMMANDS_AVAILABLE:
            base_commands.extend([
                'debug-smart-money', 'compare-strategies', 'test-smart-validation',
                'smart-money-status', 'smart-backtest', 'smart-ema-test', 'smart-macd-test'
            ])
        
        return base_commands
    
    def execute_command(self, args) -> bool:
        """Execute the specified command"""
        try:
            # Handle config check
            if args.config_check:
                return self.scanner_commands.config_check()
            
            # Scanner commands
            if args.command == 'scan':
                return self.scanner_commands.run_single_scan()
            elif args.command == 'live':
                return self.scanner_commands.run_live_scanning()
            
            # Backtest commands
            elif args.command == 'backtest':
                return self.backtest_commands.run_backtest(
                    epic=args.epic, days=args.days, show_signals=args.show_signals,
                    timeframe=args.timeframe, bb_analysis=args.bb_analysis,
                    ema_config=args.ema_config
                )
            elif args.command == 'compare-ema-configs':
                return self.backtest_commands.compare_ema_configs(
                    epic=args.epic, days=args.days, timeframe=args.timeframe,
                    ema_configs=args.ema_configs
                )
            elif args.command == 'compare-strategies':
                # Check if this should go to smart money commands or regular backtest commands
                if SMART_MONEY_COMMANDS_AVAILABLE and self.smart_money_commands and (
                    hasattr(args, 'strategies') and args.strategies and 
                    any('smart_money' in s for s in args.strategies)
                ):
                    # Route to smart money comparison
                    return self.smart_money_commands.compare_strategies(
                        epic=args.epic, days=args.days or 7
                    )
                else:
                    # Route to regular strategy comparison
                    return self.backtest_commands.compare_strategies(
                        epic=args.epic, days=args.days, timeframe=args.timeframe
                    )
            
            # Debug commands
            elif args.command == 'debug':
                if not args.epic:
                    self.logger.error("❌ --epic is required for debug command")
                    return False
                return self.debug_commands.debug_signal_detection(
                    epic=args.epic, timestamp=args.timestamp
                )
            elif args.command == 'debug-macd':
                if not args.epic:
                    self.logger.error("❌ --epic is required for debug-macd command")
                    return False
                return self.debug_commands.debug_macd_strategy(epic=args.epic)
            elif args.command == 'debug-combined':
                if not args.epic:
                    self.logger.error("❌ --epic is required for debug-combined command")
                    return False
                return self.debug_commands.debug_combined_strategies(epic=args.epic)
            elif args.command == 'test-methods':
                return self.debug_commands.test_combined_methods_exist()
            
            # BB+Supertrend commands
            elif args.command == 'debug-bb-supertrend':
                if not args.epic:
                    self.logger.error("❌ --epic is required for debug-bb-supertrend command")
                    return False
                return self._debug_bb_supertrend_strategy(args)
            elif args.command == 'test-bb-supertrend':
                if not args.epic:
                    self.logger.error("❌ --epic is required for test-bb-supertrend command")
                    return False
                return self._test_bb_supertrend_strategy(args)
            elif args.command == 'backtest-bb-supertrend':
                return self._backtest_bb_supertrend_strategy(args)
            
            # Scalping commands
            elif args.command == 'scalp':
                return self.scalping_commands.run_scalping_scan(
                    epic=args.epic, mode=args.scalping_mode or 'aggressive'
                )
            elif args.command == 'debug-scalping':
                if not args.epic:
                    self.logger.error("❌ --epic is required for debug-scalping command")
                    return False
                return self.scalping_commands.debug_scalping_signal(
                    epic=args.epic, mode=args.scalping_mode or 'aggressive'
                )
            
            # Claude commands
            elif args.command == 'test-claude':
                return self.claude_commands.test_claude_integration()
            elif args.command == 'claude-timestamp':
                if not args.epic or not args.timestamp:
                    self.logger.error("❌ --epic and --timestamp are required")
                    return False
                return self.claude_commands.analyze_timestamp_with_claude(
                    epic=args.epic, timestamp=args.timestamp, include_future=not args.no_future
                )
            elif args.command == 'claude-batch':
                return self.claude_commands.batch_analyze_backtest_signals(
                    epic=args.epic, days=args.days, max_analyses=args.max_analyses
                )
            
            # Analysis commands
            elif args.command == 'test-bb':
                return self.analysis_commands.test_bb_data(epic=args.epic or 'CS.D.EURUSD.CEEM.IP')
            elif args.command == 'compare-bb':
                return self.analysis_commands.compare_bb_filters(epic=args.epic, days=args.days)
            elif args.command == 'list-ema-configs':
                return self.analysis_commands.list_ema_configs()
            
            # KAMA commands
            elif KAMA_COMMANDS_AVAILABLE and self.kama_commands:
                if args.command == 'test-kama':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for test-kama command")
                        return False
                    return self.kama_commands.test_kama_strategy(epic=args.epic)
                elif args.command == 'debug-kama':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for debug-kama command")
                        return False
                    return self.kama_commands.debug_kama_signal(epic=args.epic)
                elif args.command == 'backtest-kama':
                    # Use the main.py backtest method for KAMA (like BB+Supertrend)
                    return self._backtest_kama_strategy(args)
                elif args.command == 'kama-signals':
                    return self.kama_commands.analyze_kama_signals(
                        epic=args.epic, days=args.days
                    )
                elif args.command == 'kama-compare':
                    return self.kama_commands.compare_kama_configs(
                        epic=args.epic, days=args.days
                    )
                elif args.command == 'kama-optimization':
                    return self.kama_commands.optimize_kama_parameters(
                        epic=args.epic, days=args.days
                    )
            
            # Dynamic config commands
            elif DYNAMIC_CONFIG_AVAILABLE and self.dynamic_config_commands:
                if args.command == 'show-configs':
                    return self.dynamic_config_commands.show_configs(
                        epic=args.epic, format='table', verbose=args.verbose
                    )
                elif args.command == 'config-performance':
                    return self.dynamic_config_commands.config_performance(
                        epic=args.epic, days=args.days or 30
                    )
                elif args.command == 'optimize-configs':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for optimize-configs command")
                        return False
                    return self.dynamic_config_commands.optimize_configs(
                        epic=args.epic, days=args.days or 30
                    )
                elif args.command == 'market-analysis':
                    return self.dynamic_config_commands.market_analysis(
                        epic=args.epic, refresh=True
                    )
                elif args.command == 'test-config-selection':
                    return self.dynamic_config_commands.test_config_selection(
                        epic=args.epic
                    )
                elif args.command == 'config-settings':
                    return self.dynamic_config_commands.settings(show_current=True)
            
            # NEW: Smart Money commands
            elif SMART_MONEY_COMMANDS_AVAILABLE and self.smart_money_commands:
                if args.command == 'debug-smart-money':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for debug-smart-money command")
                        return False
                    return self.smart_money_commands.debug_smart_money_analysis(
                        epic=args.epic, timeframe=args.timeframe
                    )
                elif args.command == 'test-smart-validation':
                    if not args.epic or not args.signal_type or args.price is None:
                        self.logger.error("❌ --epic, --signal-type, and --price are required")
                        return False
                    return self.smart_money_commands.test_smart_money_validation(
                        epic=args.epic, signal_type=args.signal_type, price=args.price
                    )
                elif args.command == 'smart-money-status':
                    return self.smart_money_commands.get_smart_money_status()
                elif args.command == 'smart-backtest':
                    return self._smart_money_backtest(args)
                elif args.command == 'smart-ema-test':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for smart-ema-test command")
                        return False
                    return self._test_smart_ema_strategy(args)
                elif args.command == 'smart-macd-test':
                    if not args.epic:
                        self.logger.error("❌ --epic is required for smart-macd-test command")
                        return False
                    return self._test_smart_macd_strategy(args)
            
            else:
                self.logger.error(f"❌ Unknown command: {args.command}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Command execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _debug_bb_supertrend_strategy(self, args) -> bool:
        """Debug Bollinger Bands + Supertrend strategy for specific epic"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            import config
            
            self.logger.info(f"🎯 Debugging BB+Supertrend Strategy for {args.epic}")
            
            # Temporarily enable BB+Supertrend strategy
            original_bb_enabled = getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False)
            config.BOLLINGER_SUPERTREND_STRATEGY = True
            
            # Set custom BB+Supertrend parameters if provided
            if args.bb_period:
                config.BB_PERIOD = args.bb_period
            if args.bb_std_dev:
                config.BB_STD_DEV = args.bb_std_dev
            if args.supertrend_period:
                config.SUPERTREND_PERIOD = args.supertrend_period
            if args.supertrend_multiplier:
                config.SUPERTREND_MULTIPLIER = args.supertrend_multiplier
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(args.epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get enhanced data
            df = detector.data_fetcher.get_enhanced_data(
                args.epic, pair, timeframe=args.timeframe, lookback_hours=48
            )
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for BB+Supertrend analysis")
                return False
            
            self.logger.info(f"✅ Data available: {len(df)} bars")
            
            # Check if BB+Supertrend strategy is available
            if not detector.bb_supertrend_strategy:
                self.logger.error("❌ BB+Supertrend strategy not initialized")
                return False
            
            # Test signal detection
            signal = detector.bb_supertrend_strategy.detect_signal(
                df, args.epic, config.SPREAD_PIPS, args.timeframe
            )
            
            if signal:
                self.logger.info(f"🎯 BB+Supertrend signal detected!")
                self.logger.info(f"   Signal Type: {signal.get('signal_type')}")
                self.logger.info(f"   Confidence: {signal.get('confidence_score', 0):.3f}")
                self.logger.info(f"   Entry Price: {signal.get('entry_price', 0):.5f}")
                self.logger.info(f"   BB Upper: {signal.get('technical_levels', {}).get('bb_upper', 'N/A')}")
                self.logger.info(f"   BB Lower: {signal.get('technical_levels', {}).get('bb_lower', 'N/A')}")
                self.logger.info(f"   Supertrend: {signal.get('technical_levels', {}).get('supertrend', 'N/A')}")
            else:
                self.logger.info("ℹ️ No BB+Supertrend signal detected")
                
                # Show current market state
                current = df.iloc[-1]
                self.logger.info(f"📊 Current Market State:")
                self.logger.info(f"   Price: {current['close']:.5f}")
                if 'bb_upper' in current:
                    self.logger.info(f"   BB Upper: {current['bb_upper']:.5f}")
                    self.logger.info(f"   BB Lower: {current['bb_lower']:.5f}")
                if 'supertrend' in current:
                    self.logger.info(f"   Supertrend: {current['supertrend']:.5f}")
                    self.logger.info(f"   ST Direction: {current.get('supertrend_direction', 'N/A')}")
            
            # Restore original settings
            config.BOLLINGER_SUPERTREND_STRATEGY = original_bb_enabled
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ BB+Supertrend debug failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_bb_supertrend_strategy(self, args) -> bool:
        """Test BB+Supertrend strategy configuration and data"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            import config
            
            self.logger.info(f"🧪 Testing BB+Supertrend Strategy Configuration")
            
            # Test configuration
            bb_period = getattr(config, 'BB_PERIOD', 20)
            bb_std_dev = getattr(config, 'BB_STD_DEV', 2.0)
            supertrend_period = getattr(config, 'SUPERTREND_PERIOD', 10)
            supertrend_multiplier = getattr(config, 'SUPERTREND_MULTIPLIER', 3.0)
            
            self.logger.info(f"📊 Configuration:")
            self.logger.info(f"   BB Period: {bb_period}")
            self.logger.info(f"   BB Std Dev: {bb_std_dev}")
            self.logger.info(f"   Supertrend Period: {supertrend_period}")
            self.logger.info(f"   Supertrend Multiplier: {supertrend_multiplier}")
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(args.epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Test data fetching
            df = detector.data_fetcher.get_enhanced_data(
                args.epic, pair, timeframe=args.timeframe, lookback_hours=72
            )
            
            if df is None:
                self.logger.error("❌ No data available")
                return False
            
            self.logger.info(f"✅ Data fetched: {len(df)} bars")
            
            # Test indicator calculations
            if 'bb_upper' not in df.columns:
                self.logger.warning("⚠️ BB indicators missing, will be calculated by strategy")
            else:
                self.logger.info("✅ BB indicators present")
            
            if 'supertrend' not in df.columns:
                self.logger.warning("⚠️ Supertrend indicators missing, will be calculated by strategy")
            else:
                self.logger.info("✅ Supertrend indicators present")
            
            # Show latest values
            current = df.iloc[-1]
            self.logger.info(f"📈 Latest Market Data:")
            self.logger.info(f"   Close: {current['close']:.5f}")
            self.logger.info(f"   High: {current['high']:.5f}")
            self.logger.info(f"   Low: {current['low']:.5f}")
            self.logger.info(f"   Volume: {current.get('volume', 'N/A')}")
            
            if 'bb_upper' in current:
                bb_width = current['bb_upper'] - current['bb_lower']
                bb_position = (current['close'] - current['bb_lower']) / bb_width if bb_width > 0 else 0.5
                self.logger.info(f"   BB Position: {bb_position:.2f} (0=lower, 1=upper)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ BB+Supertrend test failed: {e}")
            return False
    
    def _backtest_bb_supertrend_strategy(self, args) -> bool:
        """Backtest BB+Supertrend strategy"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
            import config
            
            self.logger.info(f"📈 Backtesting BB+Supertrend Strategy")
            
            # Temporarily enable only BB+Supertrend strategy
            original_strategies = {
                'ema': getattr(config, 'SIMPLE_EMA_STRATEGY', False),
                'macd': getattr(config, 'MACD_EMA_STRATEGY', False),
                'bb_st': getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False),
                'kama': getattr(config, 'KAMA_STRATEGY', False),
                'combined_mode': getattr(config, 'COMBINED_STRATEGY_MODE', None)
            }
            
            config.SIMPLE_EMA_STRATEGY = False
            config.MACD_EMA_STRATEGY = False
            config.BOLLINGER_SUPERTREND_STRATEGY = True
            config.KAMA_STRATEGY = False
            config.COMBINED_STRATEGY_MODE = None  # Disable combined mode
            
            # Set custom parameters if provided
            if args.bb_period:
                config.BB_PERIOD = args.bb_period
            if args.bb_std_dev:
                config.BB_STD_DEV = args.bb_std_dev
            if args.supertrend_period:
                config.SUPERTREND_PERIOD = args.supertrend_period
            if args.supertrend_multiplier:
                config.SUPERTREND_MULTIPLIER = args.supertrend_multiplier
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            performance_analyzer = PerformanceAnalyzer()
            
            # Set epic list
            epic_list = [args.epic] if args.epic else config.EPIC_LIST[:3]  # Limit for testing
            days = args.days or 7
            
            self.logger.info(f"🔍 Testing {len(epic_list)} epic(s) over {days} days")
            
            # Run backtest
            results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=args.timeframe
            )
            
            if not results:
                self.logger.warning("⚠️ No BB+Supertrend signals found")
            else:
                self.logger.info(f"✅ Found {len(results)} BB+Supertrend signals")
                
                # Analyze performance
                performance = performance_analyzer.analyze_performance(results)
                
                self.logger.info(f"\n📊 BB+Supertrend Strategy Performance:")
                self.logger.info(f"   Total Signals: {performance.get('total_signals', 0)}")
                self.logger.info(f"   Average Confidence: {performance.get('avg_confidence', 0):.1f}%")
                self.logger.info(f"   Strategy Distribution:")
                
                strategy_counts = {}
                for signal in results:
                    strategy = signal.get('strategy', 'unknown')
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                for strategy, count in strategy_counts.items():
                    self.logger.info(f"     {strategy}: {count} signals")
                
                if args.show_signals:
                    self.logger.info(f"\n🔍 Recent Signals:")
                    for i, signal in enumerate(results[-5:], 1):  # Show last 5
                        self.logger.info(f"   {i}. {signal.get('epic')} - {signal.get('signal_type')} "
                                       f"({signal.get('confidence_score', 0):.2f})")
            
            # Restore original strategy settings
            config.SIMPLE_EMA_STRATEGY = original_strategies['ema']
            config.MACD_EMA_STRATEGY = original_strategies['macd']
            config.BOLLINGER_SUPERTREND_STRATEGY = original_strategies['bb_st']
            config.KAMA_STRATEGY = original_strategies['kama']
            config.COMBINED_STRATEGY_MODE = original_strategies['combined_mode']
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ BB+Supertrend backtest failed: {e}")
            return False

    def _backtest_kama_strategy(self, args) -> bool:
        """Backtest KAMA strategy in isolation (similar to BB+Supertrend approach)"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
            import config
            
            self.logger.info(f"📈 Backtesting KAMA Strategy")
            
            # Store original settings
            original_strategies = {
                'ema': getattr(config, 'SIMPLE_EMA_STRATEGY', False),
                'macd': getattr(config, 'MACD_EMA_STRATEGY', False),
                'bb_st': getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False),
                'kama': getattr(config, 'KAMA_STRATEGY', False),
                'combined_mode': getattr(config, 'COMBINED_STRATEGY_MODE', None)
            }
            
            original_kama_settings = {
                'min_efficiency': getattr(config, 'KAMA_MIN_EFFICIENCY', 0.1),
                'trend_threshold': getattr(config, 'KAMA_TREND_THRESHOLD', 0.05),
                'min_bars': getattr(config, 'KAMA_MIN_BARS', 50)
            }
            
            # Force KAMA-only configuration
            config.SIMPLE_EMA_STRATEGY = False
            config.MACD_EMA_STRATEGY = False
            config.BOLLINGER_SUPERTREND_STRATEGY = False
            config.KAMA_STRATEGY = True
            config.COMBINED_STRATEGY_MODE = None  # CRITICAL: Disable combined mode
            
            # Apply custom KAMA parameters if provided
            if args.kama_min_efficiency:
                config.KAMA_MIN_EFFICIENCY = args.kama_min_efficiency
            else:
                config.KAMA_MIN_EFFICIENCY = 0.05  # Lower default for more signals
                
            if args.kama_trend_threshold:
                config.KAMA_TREND_THRESHOLD = args.kama_trend_threshold
            else:
                config.KAMA_TREND_THRESHOLD = 0.02  # Lower default for more signals
            
            # Lower minimum bars requirement
            config.KAMA_MIN_BARS = 20
            
            self.logger.info("🔧 KAMA-only configuration applied:")
            self.logger.info(f"   SIMPLE_EMA_STRATEGY: {config.SIMPLE_EMA_STRATEGY}")
            self.logger.info(f"   MACD_EMA_STRATEGY: {config.MACD_EMA_STRATEGY}")
            self.logger.info(f"   BOLLINGER_SUPERTREND_STRATEGY: {config.BOLLINGER_SUPERTREND_STRATEGY}")
            self.logger.info(f"   KAMA_STRATEGY: {config.KAMA_STRATEGY}")
            self.logger.info(f"   COMBINED_STRATEGY_MODE: {config.COMBINED_STRATEGY_MODE}")
            self.logger.info(f"   KAMA_MIN_EFFICIENCY: {config.KAMA_MIN_EFFICIENCY}")
            self.logger.info(f"   KAMA_TREND_THRESHOLD: {config.KAMA_TREND_THRESHOLD}")
            self.logger.info(f"   KAMA_MIN_BARS: {config.KAMA_MIN_BARS}")
            
            # Initialize components fresh
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            performance_analyzer = PerformanceAnalyzer()
            
            # Set epic list
            epic_list = [args.epic] if args.epic else ['CS.D.EURUSD.CEEM.IP']  # Default to EURUSD
            days = args.days or 14  # Default to 2 weeks
            timeframe = args.timeframe or '15m'
            
            self.logger.info(f"🔍 Testing {len(epic_list)} epic(s) over {days} days on {timeframe}")
            
            # Run backtest
            results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=timeframe
            )
            
            if not results:
                self.logger.warning("⚠️ No KAMA signals found")
                self.logger.info("🔍 Troubleshooting suggestions:")
                self.logger.info("   1. Try lowering --kama-min-efficiency (e.g., --kama-min-efficiency 0.01)")
                self.logger.info("   2. Try lowering --kama-trend-threshold (e.g., --kama-trend-threshold 0.001)")
                self.logger.info("   3. Try a different timeframe (e.g., --timeframe 5m)")
                self.logger.info("   4. Try more days (e.g., --days 30)")
                self.logger.info("   5. Run: python scripts/force_kama_backtest.py for manual debugging")
            else:
                self.logger.info(f"🎉 Found {len(results)} KAMA signals!")
                
                # Analyze performance
                try:
                    performance = performance_analyzer.analyze_performance(results)
                    
                    self.logger.info(f"\n📊 KAMA Strategy Performance:")
                    self.logger.info(f"   Total Signals: {performance.get('total_signals', len(results))}")
                    self.logger.info(f"   Average Confidence: {performance.get('avg_confidence', 0):.1f}%")
                    self.logger.info(f"   Signal Frequency: {len(results) / days:.2f} signals/day")
                    
                    # Strategy distribution
                    strategy_counts = {}
                    for signal in results:
                        strategy = signal.get('strategy', 'unknown')
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    self.logger.info(f"   Strategy Distribution:")
                    for strategy, count in strategy_counts.items():
                        self.logger.info(f"     {strategy}: {count} signals")
                        
                    # Signal type breakdown
                    bull_signals = sum(1 for s in results if s.get('signal_type') in ['BULL', 'BUY'])
                    bear_signals = sum(1 for s in results if s.get('signal_type') in ['BEAR', 'SELL'])
                    
                    self.logger.info(f"   Signal Types:")
                    self.logger.info(f"     Bull signals: {bull_signals}")
                    self.logger.info(f"     Bear signals: {bear_signals}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Performance analysis failed: {e}")
                
                if args.show_signals:
                    self.logger.info(f"\n🔍 Recent KAMA Signals:")
                    for i, signal in enumerate(results[-5:], 1):  # Show last 5
                        self.logger.info(f"   {i}. {signal.get('epic')} - {signal.get('signal_type')} "
                                       f"({signal.get('confidence_score', 0):.2f}) at {signal.get('timestamp')}")
                        if 'trigger_reason' in signal:
                            self.logger.info(f"      Trigger: {signal['trigger_reason']}")
                        if 'efficiency_ratio' in signal:
                            self.logger.info(f"      ER: {signal['efficiency_ratio']:.3f}")
            
            # Restore original strategy settings
            config.SIMPLE_EMA_STRATEGY = original_strategies['ema']
            config.MACD_EMA_STRATEGY = original_strategies['macd']
            config.BOLLINGER_SUPERTREND_STRATEGY = original_strategies['bb_st']
            config.KAMA_STRATEGY = original_strategies['kama']
            config.COMBINED_STRATEGY_MODE = original_strategies['combined_mode']
            
            # Restore original KAMA settings
            config.KAMA_MIN_EFFICIENCY = original_kama_settings['min_efficiency']
            config.KAMA_TREND_THRESHOLD = original_kama_settings['trend_threshold']
            config.KAMA_MIN_BARS = original_kama_settings['min_bars']
            
            return len(results) > 0
            
        except Exception as e:
            self.logger.error(f"❌ KAMA backtest failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def _debug_kama_strategy(self, args) -> bool:
        """Debug KAMA strategy for specific epic"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            import config
            
            self.logger.info(f"🔧 Debugging KAMA Strategy for {args.epic}")
            
            # Temporarily enable KAMA strategy
            original_kama_enabled = getattr(config, 'KAMA_STRATEGY', False)
            config.KAMA_STRATEGY = True
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(args.epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get enhanced data
            df = detector.data_fetcher.get_enhanced_data(
                args.epic, pair, timeframe=args.timeframe, lookback_hours=48
            )
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for KAMA analysis")
                return False
            
            self.logger.info(f"✅ Data available: {len(df)} bars")
            
            # Check if KAMA strategy is available
            if not detector.kama_strategy:
                self.logger.error("❌ KAMA strategy not initialized")
                return False
            
            # Test signal detection
            signal = detector.kama_strategy.detect_signal(
                df, args.epic, config.SPREAD_PIPS, args.timeframe
            )
            
            if signal:
                self.logger.info(f"🎯 KAMA signal detected!")
                self.logger.info(f"   Signal Type: {signal.get('signal_type')}")
                self.logger.info(f"   Confidence: {signal.get('confidence_score', 0):.3f}")
                self.logger.info(f"   Entry Price: {signal.get('entry_price', 0):.5f}")
                self.logger.info(f"   Efficiency Ratio: {signal.get('efficiency_ratio', 'N/A')}")
                self.logger.info(f"   Market Regime: {signal.get('market_regime', 'N/A')}")
            else:
                self.logger.info("ℹ️ No KAMA signal detected")
                
                # Show current market state
                current = df.iloc[-1]
                self.logger.info(f"📊 Current Market State:")
                self.logger.info(f"   Price: {current['close']:.5f}")
                if 'kama' in current:
                    self.logger.info(f"   KAMA: {current['kama']:.5f}")
                if 'efficiency_ratio' in current:
                    self.logger.info(f"   Efficiency Ratio: {current['efficiency_ratio']:.3f}")
            
            # Restore original settings
            config.KAMA_STRATEGY = original_kama_enabled
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ KAMA debug failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # NEW: Smart Money command implementations
    def _smart_money_backtest(self, args) -> bool:
        """Backtest smart money strategies"""
        try:
            from core.database import DatabaseManager
            from core.signal_detector import SignalDetector
            import config
            
            self.logger.info(f"🧠 Smart Money Strategy Backtest")
            
            # Enable smart money strategies
            original_smart_money = {
                'use_smart_ema': getattr(config, 'USE_SMART_MONEY_EMA', False),
                'use_smart_macd': getattr(config, 'USE_SMART_MONEY_MACD', False),
                'structure_validation': getattr(config, 'SMART_MONEY_STRUCTURE_VALIDATION', True),
                'order_flow_validation': getattr(config, 'SMART_MONEY_ORDER_FLOW_VALIDATION', True)
            }
            
            # Configure smart money settings based on args
            enable_structure = args.enable_structure and not args.disable_structure
            enable_order_flow = args.enable_order_flow and not args.disable_order_flow
            
            config.USE_SMART_MONEY_EMA = True
            config.USE_SMART_MONEY_MACD = True
            config.SMART_MONEY_STRUCTURE_VALIDATION = enable_structure
            config.SMART_MONEY_ORDER_FLOW_VALIDATION = enable_order_flow
            
            self.logger.info(f"🔧 Smart Money Configuration:")
            self.logger.info(f"   Structure Analysis: {'✅' if enable_structure else '❌'}")
            self.logger.info(f"   Order Flow Analysis: {'✅' if enable_order_flow else '❌'}")
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Set test parameters
            epic_list = [args.epic] if args.epic else ['CS.D.EURUSD.CEEM.IP']
            days = args.days or 7
            
            self.logger.info(f"🔍 Testing {len(epic_list)} epic(s) over {days} days")
            
            # Run backtest
            results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe=args.timeframe
            )
            
            if not results:
                self.logger.warning("⚠️ No smart money signals found")
            else:
                self.logger.info(f"🧠 Found {len(results)} smart money signals")
                
                # Analyze smart money signal distribution
                smart_signals = [s for s in results if s.get('smart_money_validated', False)]
                regular_signals = [s for s in results if not s.get('smart_money_validated', False)]
                
                self.logger.info(f"\n📊 Smart Money Analysis:")
                self.logger.info(f"   Smart Money Validated: {len(smart_signals)}")
                self.logger.info(f"   Regular Signals: {len(regular_signals)}")
                
                if smart_signals:
                    avg_smart_confidence = sum(s.get('enhanced_confidence_score', s.get('confidence_score', 0)) 
                                             for s in smart_signals) / len(smart_signals)
                    self.logger.info(f"   Average Smart Money Confidence: {avg_smart_confidence:.3f}")
            
            # Restore original settings
            config.USE_SMART_MONEY_EMA = original_smart_money['use_smart_ema']
            config.USE_SMART_MONEY_MACD = original_smart_money['use_smart_macd']
            config.SMART_MONEY_STRUCTURE_VALIDATION = original_smart_money['structure_validation']
            config.SMART_MONEY_ORDER_FLOW_VALIDATION = original_smart_money['order_flow_validation']
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart money backtest failed: {e}")
            return False
    
    def _test_smart_ema_strategy(self, args) -> bool:
        """Test Smart Money EMA strategy"""
        try:
            from core.strategies.smart_money_ema_strategy import SmartMoneyEMAStrategy
            from core.database import DatabaseManager
            from core.data_fetcher import DataFetcher
            import config
            
            self.logger.info(f"🧠 Testing Smart Money EMA Strategy for {args.epic}")
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            data_fetcher = DataFetcher(db_manager)
            
            # Create smart money EMA strategy
            smart_ema = SmartMoneyEMAStrategy(data_fetcher=data_fetcher)
            
            # Get data
            pair_info = config.PAIR_INFO.get(args.epic, {'pair': 'EURUSD'})
            df = data_fetcher.get_enhanced_data(args.epic, pair_info['pair'], args.timeframe)
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for smart EMA test")
                return False
            
            # Test signal detection
            signal = smart_ema.detect_signal(df, args.epic, config.SPREAD_PIPS, args.timeframe)
            
            if signal:
                self.logger.info(f"✅ Smart Money EMA Signal Detected!")
                self.logger.info(f"   Signal Type: {signal.get('signal_type')}")
                self.logger.info(f"   Original Confidence: {signal.get('original_confidence_score', 'N/A')}")
                self.logger.info(f"   Enhanced Confidence: {signal.get('enhanced_confidence_score', 'N/A')}")
                self.logger.info(f"   Smart Money Score: {signal.get('smart_money_score', 'N/A')}")
                
                # Show smart money analysis details
                if 'market_structure_analysis' in signal:
                    structure = signal['market_structure_analysis']
                    self.logger.info(f"   Market Structure: {structure.get('current_bias')} "
                                   f"(score: {structure.get('structure_score', 0):.3f})")
                
                if 'order_flow_analysis' in signal:
                    order_flow = signal['order_flow_analysis']
                    self.logger.info(f"   Order Flow: {order_flow.get('validation_reason', 'N/A')}")
            else:
                self.logger.info("ℹ️ No Smart Money EMA signal detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart Money EMA test failed: {e}")
            return False
    
    def _test_smart_macd_strategy(self, args) -> bool:
        """Test Smart Money MACD strategy"""
        try:
            from core.strategies.smart_money_macd_strategy import SmartMoneyMACDStrategy
            from core.database import DatabaseManager
            from core.data_fetcher import DataFetcher
            import config
            
            self.logger.info(f"📈 Testing Smart Money MACD Strategy for {args.epic}")
            
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            data_fetcher = DataFetcher(db_manager)
            
            # Create smart money MACD strategy
            smart_macd = SmartMoneyMACDStrategy()
            
            # Get data
            pair_info = config.PAIR_INFO.get(args.epic, {'pair': 'EURUSD'})
            df = data_fetcher.get_enhanced_data(args.epic, pair_info['pair'], args.timeframe)
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for smart MACD test")
                return False
            
            # Test signal detection
            signal = smart_macd.detect_signal(df, args.epic, config.SPREAD_PIPS, args.timeframe)
            
            if signal:
                self.logger.info(f"✅ Smart Money MACD Signal Detected!")
                self.logger.info(f"   Signal Type: {signal.get('signal_type')}")
                self.logger.info(f"   Original Confidence: {signal.get('original_confidence_score', 'N/A')}")
                self.logger.info(f"   Enhanced Confidence: {signal.get('enhanced_confidence_score', 'N/A')}")
                self.logger.info(f"   Order Flow Score: {signal.get('order_flow_score', 'N/A')}")
                
                # Show order flow confluence details
                if 'confluence_details' in signal:
                    self.logger.info("   Order Flow Confluences:")
                    for factor, details in signal['confluence_details'].items():
                        self.logger.info(f"     {factor}: {details.get('description', 'N/A')}")
            else:
                self.logger.info("ℹ️ No Smart Money MACD signal detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart Money MACD test failed: {e}")
            return False
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )


def main():
    """Main entry point"""
    cli = ForexScannerCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    # Setup logging
    cli.setup_logging(verbose=args.verbose)
    
    # Execute command
    success = cli.execute_command(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()