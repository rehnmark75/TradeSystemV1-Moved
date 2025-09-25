# commands/smart_money_commands.py
"""
Smart Money Debug Commands
Add these to your existing commands structure for testing and debugging smart money integration
"""

import logging
from typing import Optional, Dict
from datetime import datetime, timedelta

try:
    from core.database import DatabaseManager  
    from core.signal_detector import SignalDetector
    from core.intelligence.market_structure_analyzer import MarketStructureAnalyzer
    from core.intelligence.order_flow_analyzer import OrderFlowAnalyzer
    # Smart Money strategies removed - were experimental and not integrated
    # from core.strategies.smart_money_ema_strategy import SmartMoneyEMAStrategy
    # from core.strategies.smart_money_macd_strategy import SmartMoneyMACDStrategy
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

class SmartMoneyCommands:
    """Debug commands for smart money analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.signal_detector = SignalDetector(self.db_manager, config.USER_TIMEZONE)
            
            # Initialize smart money components (with fallback)
            try:
                self.market_structure_analyzer = MarketStructureAnalyzer()
                self.order_flow_analyzer = OrderFlowAnalyzer()
                # Smart Money strategies removed - were experimental
                self.smart_ema_strategy = None
                self.smart_macd_strategy = None
                self.logger.info("🧠 Smart Money Commands initialized with full functionality")
            except (NameError, ImportError):
                # Create mock components if smart money modules are not available
                self.market_structure_analyzer = None
                self.order_flow_analyzer = None
                self.smart_ema_strategy = None
                self.smart_macd_strategy = None
                self.logger.warning("⚠️ Smart Money Commands initialized with limited functionality (modules unavailable)")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Smart Money Commands: {e}")
            # Set all components to None for graceful degradation
            self.db_manager = None
            self.signal_detector = None
            self.market_structure_analyzer = None
            self.order_flow_analyzer = None
            self.smart_ema_strategy = None
            self.smart_macd_strategy = None
    
    def debug_smart_money_analysis(self, epic: str, timeframe: str = '5m') -> bool:
        """Debug complete smart money analysis for an epic"""
        try:
            self.logger.info(f"🧠 Starting smart money analysis debug for {epic}")
            
            # Get data
            df = self.signal_detector.data_fetcher.get_enhanced_data(epic, timeframe)
            if df is None or len(df) < 50:
                self.logger.error(f"❌ Insufficient data for {epic}")
                return False
            
            current_price = df.iloc[-1]['close']
            self.logger.info(f"📊 Data loaded: {len(df)} candles, current price: {current_price:.5f}")
            
            # 1. Market Structure Analysis
            self.logger.info("\n🏗️ === MARKET STRUCTURE ANALYSIS ===")
            try:
                structure_analysis = self.market_structure_analyzer.analyze_market_structure(
                    df, epic, timeframe
                )
                
                self.logger.info(f"Current Bias: {structure_analysis['current_bias']}")
                self.logger.info(f"Structure Score: {structure_analysis['structure_score']:.3f}")
                self.logger.info(f"Swing Points: {len(structure_analysis['swing_points'])}")
                self.logger.info(f"Structure Events: {len(structure_analysis['structure_events'])}")
                self.logger.info(f"Summary: {structure_analysis['analysis_summary']}")
                
                # Show recent swing points
                if structure_analysis['swing_points']:
                    self.logger.info("\n📍 Recent Swing Points:")
                    for sp in structure_analysis['swing_points'][-5:]:
                        self.logger.info(f"   {sp['swing_type'].upper()}: {sp['price']:.5f} "
                                       f"(strength: {sp['strength']:.3f})")
                
                # Show recent structure events
                if structure_analysis['structure_events']:
                    self.logger.info("\n🔄 Recent Structure Events:")
                    for se in structure_analysis['structure_events'][-3:]:
                        self.logger.info(f"   {se['event_type']} {se['direction']}: {se['price']:.5f} "
                                       f"(confidence: {se['confidence']:.3f})")
                
            except Exception as e:
                self.logger.error(f"❌ Market structure analysis failed: {e}")
            
            # 2. Order Flow Analysis
            self.logger.info("\n📊 === ORDER FLOW ANALYSIS ===")
            try:
                order_flow_analysis = self.order_flow_analyzer.analyze_order_flow(
                    df, epic, timeframe
                )
                
                self.logger.info(f"Order Flow Bias: {order_flow_analysis['order_flow_bias']}")
                self.logger.info(f"Order Blocks: {len(order_flow_analysis['order_blocks'])}")
                self.logger.info(f"Fair Value Gaps: {len(order_flow_analysis['fair_value_gaps'])}")
                self.logger.info(f"Supply/Demand Zones: {len(order_flow_analysis['supply_demand_zones'])}")
                self.logger.info(f"Summary: {order_flow_analysis['analysis_summary']}")
                
                # Show active order blocks
                if order_flow_analysis['order_blocks']:
                    self.logger.info("\n🏦 Active Order Blocks:")
                    for ob in order_flow_analysis['order_blocks'][-3:]:
                        distance = abs(current_price - ob['high']) if 'BEARISH' in ob['block_type'] else abs(current_price - ob['low'])
                        self.logger.info(f"   {ob['block_type']}: {ob['low']:.5f}-{ob['high']:.5f} "
                                       f"(strength: {ob['strength']:.3f}, distance: {distance*10000:.1f} pips)")
                
                # Show unfilled FVGs
                unfilled_fvgs = [fvg for fvg in order_flow_analysis['fair_value_gaps'] 
                               if fvg['filled_percentage'] < 50]
                if unfilled_fvgs:
                    self.logger.info("\n🕳️ Unfilled Fair Value Gaps:")
                    for fvg in unfilled_fvgs[-3:]:
                        self.logger.info(f"   {fvg['gap_type']}: {fvg['bottom']:.5f}-{fvg['top']:.5f} "
                                       f"({fvg['size_pips']:.1f} pips, {fvg['filled_percentage']:.1f}% filled)")
                
            except Exception as e:
                self.logger.error(f"❌ Order flow analysis failed: {e}")
            
            # 3. Smart Money EMA Strategy Test
            self.logger.info("\n🎯 === SMART MONEY EMA STRATEGY TEST ===")
            try:
                smart_ema_signal = self.smart_ema_strategy.detect_signal(
                    df, epic, config.SPREAD_PIPS, timeframe
                )
                
                if smart_ema_signal:
                    self.logger.info(f"✅ Smart EMA Signal Detected!")
                    self.logger.info(f"   Signal Type: {smart_ema_signal['signal_type']}")
                    self.logger.info(f"   Entry Price: {smart_ema_signal['entry_price']:.5f}")
                    self.logger.info(f"   Original Confidence: {smart_ema_signal.get('original_confidence_score', 'N/A')}")
                    self.logger.info(f"   Enhanced Confidence: {smart_ema_signal.get('enhanced_confidence_score', smart_ema_signal['confidence_score']):.3f}")
                    self.logger.info(f"   Smart Money Score: {smart_ema_signal.get('smart_money_score', 'N/A')}")
                    
                    # Show smart money analysis
                    if 'market_structure_analysis' in smart_ema_signal:
                        structure = smart_ema_signal['market_structure_analysis']
                        self.logger.info(f"   Structure Aligned: {structure['structure_aligned']}")
                        self.logger.info(f"   Structure Score: {structure['structure_score']:.3f}")
                    
                    if 'order_flow_analysis' in smart_ema_signal:
                        order_flow = smart_ema_signal['order_flow_analysis']
                        self.logger.info(f"   Order Flow Aligned: {order_flow['order_flow_aligned']}")
                        self.logger.info(f"   Order Flow Score: {order_flow['order_flow_score']:.3f}")
                else:
                    self.logger.info("❌ No Smart EMA signal detected")
                    
            except Exception as e:
                self.logger.error(f"❌ Smart EMA strategy test failed: {e}")
            
            # 4. Smart Money MACD Strategy Test
            self.logger.info("\n🎯 === SMART MONEY MACD STRATEGY TEST ===")
            try:
                smart_macd_signal = self.smart_macd_strategy.detect_signal(
                    df, epic, config.SPREAD_PIPS, timeframe
                )
                
                if smart_macd_signal:
                    self.logger.info(f"✅ Smart MACD Signal Detected!")
                    self.logger.info(f"   Signal Type: {smart_macd_signal['signal_type']}")
                    self.logger.info(f"   Entry Price: {smart_macd_signal['entry_price']:.5f}")
                    self.logger.info(f"   Original Confidence: {smart_macd_signal.get('original_confidence_score', 'N/A')}")
                    self.logger.info(f"   Enhanced Confidence: {smart_macd_signal.get('enhanced_confidence_score', smart_macd_signal['confidence_score']):.3f}")
                    self.logger.info(f"   Order Flow Score: {smart_macd_signal.get('order_flow_score', 'N/A')}")
                    
                    # Show order flow confluence
                    if 'confluence_details' in smart_macd_signal:
                        self.logger.info("   Order Flow Confluences:")
                        for factor, details in smart_macd_signal['confluence_details'].items():
                            self.logger.info(f"     {factor}: {details['description']} (score: {details['score']:.3f})")
                else:
                    self.logger.info("❌ No Smart MACD signal detected")
                    
            except Exception as e:
                self.logger.error(f"❌ Smart MACD strategy test failed: {e}")
            
            self.logger.info("\n🎯 Smart money analysis debug completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart money debug failed: {e}")
            return False
    
    def compare_strategies(self, epic: str, days: int = 7) -> bool:
        """Compare regular vs smart money strategies"""
        try:
            self.logger.info(f"🔄 Comparing strategies for {epic} over {days} days")
            
            # This would integrate with your existing backtest engine
            # For now, just show the framework
            
            results = {
                'regular_ema': {'signals': 0, 'wins': 0, 'avg_profit': 0},
                'smart_ema': {'signals': 0, 'wins': 0, 'avg_profit': 0},
                'regular_macd': {'signals': 0, 'wins': 0, 'avg_profit': 0},
                'smart_macd': {'signals': 0, 'wins': 0, 'avg_profit': 0}
            }
            
            # TODO: Implement actual comparison logic using your backtest engine
            
            self.logger.info("📊 Strategy Comparison Results:")
            for strategy, metrics in results.items():
                self.logger.info(f"   {strategy}: {metrics['signals']} signals, "
                               f"{metrics['wins']} wins, {metrics['avg_profit']:.1f} avg profit")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Strategy comparison failed: {e}")
            return False
    
    def test_smart_money_validation(self, epic: str, signal_type: str, price: float) -> bool:
        """Test smart money validation for a hypothetical signal"""
        try:
            self.logger.info(f"🧪 Testing smart money validation for {signal_type} at {price:.5f}")
            
            # Test market structure validation
            structure_validation = self.market_structure_analyzer.validate_signal_against_structure(
                signal_type, price, epic
            )
            
            self.logger.info("🏗️ Market Structure Validation:")
            self.logger.info(f"   Aligned: {structure_validation['structure_aligned']}")
            self.logger.info(f"   Score: {structure_validation['structure_score']:.3f}")
            self.logger.info(f"   Reason: {structure_validation['validation_reason']}")
            self.logger.info(f"   Action: {structure_validation['recommended_action']}")
            
            # Test order flow validation
            order_flow_validation = self.order_flow_analyzer.validate_signal_against_order_flow(
                signal_type, price, epic
            )
            
            self.logger.info("\n📊 Order Flow Validation:")
            self.logger.info(f"   Aligned: {order_flow_validation['order_flow_aligned']}")
            self.logger.info(f"   Score: {order_flow_validation['order_flow_score']:.3f}")
            self.logger.info(f"   Reason: {order_flow_validation['validation_reason']}")
            self.logger.info(f"   Action: {order_flow_validation['recommended_action']}")
            
            if order_flow_validation.get('nearest_levels'):
                levels = order_flow_validation['nearest_levels']
                self.logger.info("   Nearest Levels:")
                if levels.get('nearest_support'):
                    self.logger.info(f"     Support: {levels['nearest_support']['price']:.5f} "
                                   f"({levels['nearest_support']['type']})")
                if levels.get('nearest_resistance'):
                    self.logger.info(f"     Resistance: {levels['nearest_resistance']['price']:.5f} "
                                   f"({levels['nearest_resistance']['type']})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Smart money validation test failed: {e}")
            return False
    
    def get_smart_money_status(self) -> Dict:
        """Get overall smart money system status"""
        try:
            status = {
                'market_structure_analyzer': {
                    'initialized': self.market_structure_analyzer is not None,
                    'swing_lookback': getattr(config, 'STRUCTURE_SWING_LOOKBACK', 5),
                    'bos_confirmation_pips': getattr(config, 'STRUCTURE_BOS_CONFIRMATION_PIPS', 5)
                },
                'order_flow_analyzer': {
                    'initialized': self.order_flow_analyzer is not None,
                    'min_ob_size_pips': getattr(config, 'ORDER_FLOW_MIN_OB_SIZE_PIPS', 8),
                    'min_fvg_size_pips': getattr(config, 'ORDER_FLOW_MIN_FVG_SIZE_PIPS', 5)
                },
                'smart_strategies': {
                    'ema_initialized': self.smart_ema_strategy is not None,
                    'macd_initialized': self.smart_macd_strategy is not None,
                    'ema_status': self.smart_ema_strategy.get_smart_money_status() if self.smart_ema_strategy else None,
                    'macd_status': self.smart_macd_strategy.get_smart_money_status() if self.smart_macd_strategy else None
                },
                'configuration': {
                    'structure_validation': getattr(config, 'SMART_MONEY_STRUCTURE_VALIDATION', True),
                    'order_flow_validation': getattr(config, 'SMART_MONEY_ORDER_FLOW_VALIDATION', True),
                    'structure_weight': getattr(config, 'SMART_MONEY_STRUCTURE_WEIGHT', 0.3),
                    'order_flow_weight': getattr(config, 'SMART_MONEY_ORDER_FLOW_WEIGHT', 0.2)
                }
            }
            
            self.logger.info("🧠 Smart Money System Status:")
            for component, details in status.items():
                self.logger.info(f"   {component}: {details}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Smart money status check failed: {e}")
            return {}


# Add these functions to your main.py CLI interface:

def add_smart_money_commands_to_main():
    """
    Add these command handlers to your main.py file:
    """
    
    # In your main.py, add these imports
    from commands.smart_money_commands import SmartMoneyCommands
    
    # Add these command handlers to your CLI parser
    
    def handle_debug_smart_money(args):
        """Debug smart money analysis for specific epic"""
        commands = SmartMoneyCommands()
        return commands.debug_smart_money_analysis(
            args.epic, 
            getattr(args, 'timeframe', '5m')
        )
    
    def handle_compare_strategies(args):
        """Compare regular vs smart money strategies"""
        commands = SmartMoneyCommands()
        return commands.compare_strategies(
            args.epic,
            getattr(args, 'days', 7)
        )
    
    def handle_test_smart_validation(args):
        """Test smart money validation"""
        commands = SmartMoneyCommands()
        return commands.test_smart_money_validation(
            args.epic,
            args.signal_type,
            float(args.price)
        )
    
    def handle_smart_money_status(args):
        """Get smart money system status"""
        commands = SmartMoneyCommands()
        status = commands.get_smart_money_status()
        return len(status) > 0
    
    # Add these to your argument parser:
    """
    # Smart money debug commands
    parser_debug_smart = subparsers.add_parser('debug-smart-money', help='Debug smart money analysis')
    parser_debug_smart.add_argument('--epic', required=True, help='Epic to analyze')
    parser_debug_smart.add_argument('--timeframe', default='5m', help='Timeframe')
    parser_debug_smart.set_defaults(func=handle_debug_smart_money)
    
    parser_compare = subparsers.add_parser('compare-strategies', help='Compare regular vs smart strategies')
    parser_compare.add_argument('--epic', required=True, help='Epic to compare')
    parser_compare.add_argument('--days', type=int, default=7, help='Days to compare')
    parser_compare.set_defaults(func=handle_compare_strategies)
    
    parser_test_validation = subparsers.add_parser('test-smart-validation', help='Test smart money validation')
    parser_test_validation.add_argument('--epic', required=True, help='Epic')
    parser_test_validation.add_argument('--signal-type', required=True, choices=['BUY', 'SELL', 'BULL', 'BEAR'], help='Signal type')
    parser_test_validation.add_argument('--price', required=True, help='Price to test')
    parser_test_validation.set_defaults(func=handle_test_smart_validation)
    
    parser_smart_status = subparsers.add_parser('smart-money-status', help='Get smart money system status')
    parser_smart_status.set_defaults(func=handle_smart_money_status)
    """