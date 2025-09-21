# commands/claude_commands.py
"""
Claude Commands Module
Handles Claude API integration and AI-powered analysis
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    import config
    import pandas as pd
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

class ClaudeCommands:
    """Claude API command implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            from alerts.validation.timestamp_validator import TimestampValidator
        except ImportError:
            try:
                from forex_scanner.alerts.validation.timestamp_validator import TimestampValidator
            except ImportError:
                # Create a mock validator if not available
                class TimestampValidator:
                    def __init__(self):
                        pass
                    def validate(self, *args, **kwargs):
                        return True
        self.timestamp_validator = TimestampValidator()

    def test_claude_integration(self) -> bool:
        """Test Claude API integration"""
        self.logger.info("🤖 Testing Claude API integration")
        
        if not config.CLAUDE_API_KEY:
            self.logger.error("❌ CLAUDE_API_KEY not configured")
            return False
        
        try:
            from alerts import ClaudeAnalyzer
            
            analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
            
            # Test with dummy signal
            test_signal = {
                'signal_type': 'BULL',
                'epic': 'CS.D.EURUSD.CEEM.IP',
                'price': 1.0850,
                'timestamp': '2025-01-01 12:00:00',
                'confidence_score': 0.75,
                'ema_9': 1.0845,
                'ema_21': 1.0840,
                'ema_200': 1.0830
            }
            
            analysis = analyzer.analyze_signal(test_signal)
            if analysis:
                self.logger.info("✅ Claude API test successful")
                self.logger.info(f"Sample analysis: {analysis[:100]}...")
                return True
            else:
                self.logger.error("❌ Claude API returned empty response")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Claude API test failed: {e}")
            return False
    
    def analyze_timestamp_with_claude(
        self, 
        epic: str, 
        timestamp: str, 
        include_future: bool = True
    ) -> bool:
        """Analyze a specific timestamp with Claude for system fine-tuning"""
        self.logger.info(f"🕐 Analyzing {epic} at {timestamp} with Claude")
        
        try:
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Check if Claude is available
            if not config.CLAUDE_API_KEY:
                self.logger.error("❌ CLAUDE_API_KEY not configured")
                return False
            
            from alerts import ClaudeAnalyzer
            claude_analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
            
            # Get Claude's analysis
            analysis = claude_analyzer.analyze_signal_at_timestamp(
                epic=epic,
                timestamp_str=timestamp,
                signal_detector=detector,
                include_future_analysis=include_future
            )
            
            if 'error' in analysis:
                self.logger.error(f"❌ Analysis failed: {analysis['error']}")
                return False
            
            # Display comprehensive results
            self._display_claude_analysis_results(analysis, include_future)
            
            # Save detailed analysis to file
            self._save_analysis_to_file(analysis, epic, timestamp)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Timestamp analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def batch_analyze_backtest_signals(
        self, 
        epic: str = None, 
        days: int = 7, 
        max_analyses: int = 5
    ) -> bool:
        """Run backtest and analyze the most interesting signals with Claude"""
        self.logger.info(f"🔍 Running batch analysis of backtest signals")
        
        try:
            # Run backtest first
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            epic_list = [epic] if epic else config.EPIC_LIST
            
            # Get backtest results
            results = detector.backtest_signals(
                epic_list=epic_list,
                lookback_days=days,
                use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                spread_pips=config.SPREAD_PIPS,
                timeframe='5m'
            )
            
            if not results:
                self.logger.info("No backtest signals found")
                return False
            
            self.logger.info(f"Found {len(results)} backtest signals")
            
            # Select most interesting signals for analysis
            signals_to_analyze = self._select_interesting_signals(results, max_analyses)
            
            self.logger.info(f"🤖 Analyzing {len(signals_to_analyze)} signals with Claude...")
            
            # Initialize Claude
            if not config.CLAUDE_API_KEY:
                self.logger.error("❌ CLAUDE_API_KEY not configured")
                return False
            
            from alerts import ClaudeAnalyzer
            claude_analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
            
            # Analyze each signal
            successful_analyses = 0
            for i, signal in enumerate(signals_to_analyze, 1):
                self.logger.info(f"\n📊 Analyzing signal {i}/{len(signals_to_analyze)}: {signal['epic']} {signal['signal_type']}")
                
                timestamp_str = self._format_signal_timestamp(signal['timestamp'])
                
                analysis = claude_analyzer.analyze_signal_at_timestamp(
                    epic=signal['epic'],
                    timestamp_str=timestamp_str,
                    signal_detector=detector,
                    include_future_analysis=True
                )
                
                if analysis and 'error' not in analysis:
                    # Quick summary
                    self._display_batch_analysis_summary(analysis, signal)
                    successful_analyses += 1
                else:
                    self.logger.warning(f"  ❌ Analysis failed for this signal")
                
                # Small delay between analyses
                import time
                time.sleep(2)
            
            self.logger.info(f"✅ Batch analysis complete: {successful_analyses}/{len(signals_to_analyze)} successful")
            self.logger.info(f"📁 Check claude_analysis/ folder for detailed reports")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch analysis failed: {e}")
            return False
    
    def get_market_overview(self, epic_list: List[str] = None) -> bool:
        """Get general market overview from Claude"""
        self.logger.info("🌍 Getting market overview from Claude")
        
        if not config.CLAUDE_API_KEY:
            self.logger.error("❌ CLAUDE_API_KEY not configured")
            return False
        
        try:
            from alerts import ClaudeAnalyzer
            claude_analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
            
            epic_list = epic_list or config.EPIC_LIST
            overview = claude_analyzer.get_market_overview(epic_list)
            
            if overview:
                self.logger.info("📊 MARKET OVERVIEW FROM CLAUDE:")
                self.logger.info("=" * 60)
                self.logger.info(overview)
                self.logger.info("=" * 60)
                return True
            else:
                self.logger.error("❌ Failed to get market overview")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Market overview failed: {e}")
            return False
    
    def _display_claude_analysis_results(self, analysis: Dict, include_future: bool):
        """
        FIXED: Display comprehensive Claude analysis results with proper key handling
        """
        self.logger.info("🔬 CLAUDE TIMESTAMP ANALYSIS RESULTS:")
        self.logger.info("=" * 80)
        
        # FIXED: Use actual keys from the analysis response
        # Basic info - check if keys exist before accessing
        if 'epic' in analysis:
            self.logger.info(f"Epic: {analysis['epic']}")
        
        # Use the correct timestamp keys
        if 'timestamp_analyzed' in analysis:
            self.logger.info(f"Timestamp Analyzed: {analysis['timestamp_analyzed']}")
        elif 'actual_candle_time' in analysis:
            self.logger.info(f"Candle Time: {analysis['actual_candle_time']}")
        
        # FIXED: Use correct analysis result keys
        if 'score' in analysis:
            self.logger.info(f"Claude Quality Score: {analysis['score']}/10")
        
        if 'decision' in analysis:
            decision_icon = "✅" if analysis['decision'] == 'APPROVE' else "❌" if analysis['decision'] == 'REJECT' else "⚠️"
            self.logger.info(f"Claude Decision: {decision_icon} {analysis['decision']}")
        
        if 'approved' in analysis:
            approval_icon = "✅" if analysis['approved'] else "❌"
            self.logger.info(f"Claude Approval: {approval_icon} {analysis['approved']}")
        
        if 'reason' in analysis:
            self.logger.info(f"Reasoning: {analysis['reason']}")
        
        # Analysis mode and technical validation
        if 'mode' in analysis:
            self.logger.info(f"Analysis Mode: {analysis['mode']}")
        
        if 'technical_validation_passed' in analysis:
            validation_icon = "✅" if analysis['technical_validation_passed'] else "❌"
            self.logger.info(f"Technical Validation: {validation_icon} {analysis['technical_validation_passed']}")
        
        # Market data information
        if 'market_data' in analysis:
            market_data = analysis['market_data']
            self.logger.info(f"\n📊 MARKET DATA:")
            self.logger.info(f"Price: {market_data.get('price', 'N/A')}")
            if 'volume' in market_data:
                self.logger.info(f"Volume: {market_data['volume']}")
        
        # Pair and timestamp method info
        if 'pair' in analysis:
            self.logger.info(f"Pair: {analysis['pair']}")
        
        if 'timestamp_method' in analysis:
            self.logger.info(f"Timestamp Method: {analysis['timestamp_method']}")
        
        # Future analysis and outcome (if available and included)
        if include_future and 'outcome' in analysis:
            self._display_future_outcome(analysis['outcome'])
        
        if include_future and 'outcome_accuracy' in analysis:
            accuracy_icon = "✅" if analysis['outcome_accuracy'] == 'correct' else "❌" if analysis['outcome_accuracy'] == 'incorrect' else "⚠️"
            self.logger.info(f"Outcome Accuracy: {accuracy_icon} {analysis['outcome_accuracy']}")
        
        # Raw Claude response
        if 'raw_response' in analysis:
            self.logger.info(f"\n🤖 CLAUDE'S FULL ANALYSIS:")
            self.logger.info("-" * 80)
            # Truncate if very long
            raw_response = analysis['raw_response']
            if len(raw_response) > 1000:
                self.logger.info(f"{raw_response[:1000]}...")
                self.logger.info(f"[Response truncated - full analysis saved to file]")
            else:
                self.logger.info(raw_response)
        
        self.logger.info("=" * 80)

    def _display_future_outcome(self, outcome: Dict):
        """
        FIXED: Display future outcome data with proper key handling
        """
        self.logger.info(f"\n📊 ACTUAL FUTURE OUTCOME:")
        
        # Handle different possible outcome structures
        if 'max_gain_pips' in outcome:
            self.logger.info(f"Max Gain: {outcome['max_gain_pips']:.1f} pips")
        
        if 'max_loss_pips' in outcome:
            self.logger.info(f"Max Loss: {outcome['max_loss_pips']:.1f} pips")
        
        if 'net_movement_pips' in outcome:
            self.logger.info(f"Net Movement: {outcome['net_movement_pips']:.1f} pips")
        
        if 'favorable_movement' in outcome:
            favorable_icon = "✅" if outcome['favorable_movement'] else "❌"
            self.logger.info(f"Favorable Movement: {favorable_icon} {outcome['favorable_movement']}")
        
        if 'next_hour_high' in outcome and 'next_hour_low' in outcome:
            price_range = outcome['next_hour_high'] - outcome['next_hour_low']
            self.logger.info(f"Price Range: {outcome['next_hour_low']:.5f} - {outcome['next_hour_high']:.5f}")
            
            if 'price_range_pips' in outcome:
                self.logger.info(f"Range in Pips: {outcome['price_range_pips']:.1f}")
        
        if 'candles_analyzed' in outcome:
            self.logger.info(f"Future Candles Analyzed: {outcome['candles_analyzed']}")

    
    def _display_future_performance(self, perf: Dict):
        """Display future performance data"""
        self.logger.info(f"\n📊 ACTUAL OUTCOME:")
        self.logger.info(f"Max Bull Profit: {perf['max_profit_bull_pips']:.1f} pips")
        self.logger.info(f"Max Bull Loss: {perf['max_loss_bull_pips']:.1f} pips")
        self.logger.info(f"Max Bear Profit: {perf['max_profit_bear_pips']:.1f} pips")
        self.logger.info(f"Max Bear Loss: {perf['max_loss_bear_pips']:.1f} pips")
        
        # Timing info
        bars_to_20_up = perf.get('bars_to_20_pips_up', 'Never')
        bars_to_20_down = perf.get('bars_to_20_pips_down', 'Never')
        self.logger.info(f"20 pips target timing: ↑{bars_to_20_up} bars / ↓{bars_to_20_down} bars")
    
    def _display_batch_analysis_summary(self, analysis: Dict, signal: Dict):
        """
        FIXED: Display quick summary for batch analysis with proper key handling
        """
        # Use the correct keys from the new Claude analyzer
        score = analysis.get('score', 'N/A')
        decision = analysis.get('decision', 'N/A')
        approved = analysis.get('approved', False)
        
        # Status icon based on decision
        if decision == 'APPROVE':
            status_icon = "✅"
        elif decision == 'REJECT':
            status_icon = "❌"
        else:
            status_icon = "⚠️"
        
        self.logger.info(f"  {status_icon} Claude Score: {score}/10, Decision: {decision}, Approved: {approved}")
        
        # Show reasoning if available
        if 'reason' in analysis:
            reason = analysis['reason']
            # Truncate long reasons
            if len(reason) > 100:
                reason = reason[:100] + "..."
            self.logger.info(f"  Reasoning: {reason}")
        
        # Show outcome if available
        if 'outcome' in analysis:
            outcome = analysis['outcome']
            if 'max_gain_pips' in outcome and 'max_loss_pips' in outcome:
                self.logger.info(f"  Future outcome: +{outcome['max_gain_pips']:.1f} / -{outcome['max_loss_pips']:.1f} pips")
            
            if 'outcome_accuracy' in analysis:
                accuracy = analysis['outcome_accuracy']
                accuracy_icon = "✅" if accuracy == 'correct' else "❌" if accuracy == 'incorrect' else "⚠️"
                self.logger.info(f"  Prediction accuracy: {accuracy_icon} {accuracy}")


    
    def _select_interesting_signals(self, results: List[Dict], max_analyses: int) -> List[Dict]:
        """Select the most interesting signals for analysis"""
        # Sort by confidence and select most interesting ones
        sorted_signals = sorted(results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        signals_to_analyze = []
        
        # Add top confidence signals
        signals_to_analyze.extend(sorted_signals[:max_analyses//2])
        
        # Add signals with interesting performance patterns
        performance_signals = [s for s in results if 'max_profit_pips' in s]
        if performance_signals:
            # Sort by risk/reward ratio
            performance_signals.sort(key=lambda x: x.get('risk_reward_ratio', 0), reverse=True)
            signals_to_analyze.extend(performance_signals[:max_analyses//2])
        
        # Remove duplicates and limit
        unique_signals = []
        seen = set()
        for signal in signals_to_analyze:
            key = f"{signal['epic']}_{signal['timestamp']}"
            if key not in seen:
                unique_signals.append(signal)
                seen.add(key)
        
        return unique_signals[:max_analyses]
    
    def _format_signal_timestamp(self, timestamp) -> str:
        """Format signal timestamp for Claude analysis"""
        if hasattr(timestamp, 'strftime'):
            return timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            return str(timestamp)[:16]
    
    def _save_analysis_to_file(self, analysis: Dict, epic: str, timestamp: str):
        """
        FIXED: Save Claude analysis to file with proper key handling
        """
        try:
            import os
            from datetime import datetime
            
            # Create analysis directory if it doesn't exist
            analysis_dir = "claude_analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Create safe filename
            safe_timestamp = timestamp.replace(":", "-").replace(" ", "_").replace("T", "_")
            safe_epic = epic.replace(".", "_")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{analysis_dir}/analysis_{safe_epic}_{safe_timestamp}_{current_time}.txt"
            
            # Write comprehensive analysis
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Timestamp Analysis Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 80 + "\n\n")
                
                # Basic information
                f.write(f"SIGNAL INFORMATION:\n")
                f.write(f"Epic: {analysis.get('epic', epic)}\n")
                f.write(f"Timestamp Analyzed: {analysis.get('timestamp_analyzed', timestamp)}\n")
                f.write(f"Actual Candle Time: {analysis.get('actual_candle_time', 'N/A')}\n")
                f.write(f"Pair: {analysis.get('pair', 'N/A')}\n")
                f.write(f"Timestamp Method: {analysis.get('timestamp_method', 'N/A')}\n")
                f.write(f"\n")
                
                # Claude's analysis results
                f.write(f"CLAUDE ANALYSIS RESULTS:\n")
                f.write(f"Score: {analysis.get('score', 'N/A')}/10\n")
                f.write(f"Decision: {analysis.get('decision', 'N/A')}\n")
                f.write(f"Approved: {analysis.get('approved', 'N/A')}\n")
                f.write(f"Analysis Mode: {analysis.get('mode', 'N/A')}\n")
                f.write(f"Technical Validation: {analysis.get('technical_validation_passed', 'N/A')}\n")
                f.write(f"Reasoning: {analysis.get('reason', 'N/A')}\n")
                f.write(f"\n")
                
                # Market data
                if 'market_data' in analysis:
                    f.write(f"MARKET DATA:\n")
                    market_data = analysis['market_data']
                    for key, value in market_data.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"\n")
                
                # Future outcome
                if 'outcome' in analysis:
                    f.write(f"FUTURE OUTCOME:\n")
                    outcome = analysis['outcome']
                    for key, value in outcome.items():
                        f.write(f"{key}: {value}\n")
                    f.write(f"Outcome Accuracy: {analysis.get('outcome_accuracy', 'N/A')}\n")
                    f.write(f"\n")
                
                # Full Claude response
                f.write(f"FULL CLAUDE RESPONSE:\n")
                f.write(f"=" * 80 + "\n")
                f.write(analysis.get('raw_response', 'No raw response available'))
                f.write(f"\n" + "=" * 80 + "\n")
                
                # Analysis timestamp
                if 'analysis_timestamp' in analysis:
                    f.write(f"\nAnalysis Timestamp: {analysis['analysis_timestamp']}\n")
            
            self.logger.info(f"📁 Analysis saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis: {e}")
            import traceback
            self.logger.debug(f"Save error traceback: {traceback.format_exc()}")
    
    def test_claude_connection(self) -> bool:
        """Test basic Claude connection without full analysis"""
        if not config.CLAUDE_API_KEY:
            self.logger.error("❌ No Claude API key configured")
            return False
        
        try:
            from alerts import ClaudeAnalyzer
            analyzer = ClaudeAnalyzer(config.CLAUDE_API_KEY)
            
            success = analyzer.test_connection()
            if success:
                self.logger.info("✅ Claude connection test passed")
            else:
                self.logger.error("❌ Claude connection test failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Claude connection test error: {e}")
            return False