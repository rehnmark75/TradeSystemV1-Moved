# commands/scan_commands.py - Fixed scanner initialization
"""
Enhanced Scan Commands with KAMA Strategy Integration
Handles live scanning and signal detection across all strategies
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
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



class ScanCommands:
    """Enhanced scan command implementations with KAMA support"""
    
    def __init__(self, scanner=None):
        # Note: scanner parameter is optional for compatibility
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components directly
        self.db_manager = DatabaseManager()
        self.data_fetcher = DataFetcher(self.db_manager)
        self.signal_detector = SignalDetector()
        
        # Initialize Claude API and notifications
        api_key = getattr(config, 'CLAUDE_API_KEY', None)
        self.claude_analyzer = ClaudeAnalyzer(api_key) if api_key else None
        self.notification_manager = NotificationManager()
        
        # Track scanning statistics
        self.scan_stats = {
            'total_scans': 0,
            'total_signals': 0,
            'strategy_signals': {},
            'last_scan_time': None
        }
    
    def start_live_scanning(
        self,
        pairs: List[str] = None,
        timeframe: str = '5m',
        strategy: str = 'combined',
        scan_interval: int = 60,
        enable_claude_analysis: bool = False
    ):
        """
        Start live signal scanning with enhanced strategy support
        
        Args:
            pairs: List of trading pairs to scan
            timeframe: Timeframe for scanning
            strategy: Strategy to use (ema/macd/kama/combined/scalping_*)
            scan_interval: Scan interval in seconds
            enable_claude_analysis: Enable Claude AI analysis
        """
        if pairs is None:
            pairs = getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP'])
        
        self.logger.info(f"🚀 Starting live scanning")
        self.logger.info(f"   Pairs: {len(pairs)} ({', '.join(pairs[:3])}{'...' if len(pairs) > 3 else ''})")
        self.logger.info(f"   Strategy: {strategy}")
        self.logger.info(f"   Timeframe: {timeframe}")
        self.logger.info(f"   Interval: {scan_interval}s")
        self.logger.info(f"   Claude analysis: {'enabled' if enable_claude_analysis else 'disabled'}")
        
        try:
            while True:
                scan_start = time.time()
                signals_found = []
                
                # Scan each pair
                for pair in pairs:
                    try:
                        pair_signals = self._scan_single_pair(
                            pair, timeframe, strategy, enable_claude_analysis
                        )
                        if pair_signals:
                            signals_found.extend(pair_signals)
                    
                    except Exception as e:
                        self.logger.error(f"❌ Error scanning {pair}: {e}")
                        continue
                
                # Update statistics
                self._update_scan_stats(signals_found, strategy)
                
                # Display scan summary
                scan_duration = time.time() - scan_start
                self._display_scan_summary(signals_found, scan_duration, strategy)
                
                # Send notifications for new signals
                if signals_found:
                    self._send_signal_notifications(signals_found)
                
                # Wait for next scan
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Live scanning stopped by user")
            self._display_final_stats()
        
        except Exception as e:
            self.logger.error(f"💥 Fatal error in live scanning: {e}")
            raise
    
    def scan_all_pairs(
        self,
        pairs: List[str] = None,
        timeframe: str = '5m',
        strategy: str = 'combined',
        show_confluence: bool = False
    ) -> List[Dict]:
        """
        Scan all pairs once and return results
        
        Args:
            pairs: List of trading pairs to scan
            timeframe: Timeframe for scanning
            strategy: Strategy to use
            show_confluence: Show confluence analysis
            
        Returns:
            List of detected signals
        """
        if pairs is None:
            pairs = getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP'])
        
        self.logger.info(f"🔍 Scanning {len(pairs)} pairs with {strategy} strategy")
        
        all_signals = []
        confluence_results = []
        
        try:
            for pair in pairs:
                try:
                    # Get signals for this pair
                    pair_signals = self._scan_single_pair(pair, timeframe, strategy, False)
                    if pair_signals:
                        all_signals.extend(pair_signals)
                    
                    # Get confluence analysis if requested
                    if show_confluence:
                        confluence = self._get_confluence_analysis(pair, timeframe)
                        if confluence:
                            confluence_results.append(confluence)
                
                except Exception as e:
                    self.logger.error(f"❌ Error scanning {pair}: {e}")
                    continue
            
            # Display results
            self._display_scan_results(all_signals, confluence_results, strategy)
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch scanning: {e}")
            return []
    
    def scan_kama_efficiency_analysis(
        self,
        pairs: List[str] = None,
        timeframe: str = '5m',
        min_efficiency_ratio: float = 0.5
    ) -> Dict:
        """
        Scan pairs for KAMA efficiency analysis to identify trending markets
        
        Args:
            pairs: List of trading pairs to analyze
            timeframe: Timeframe for analysis
            min_efficiency_ratio: Minimum ER for trending market classification
            
        Returns:
            Dictionary with efficiency analysis results
        """
        if pairs is None:
            pairs = getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP'])
        
        self.logger.info(f"🔄 KAMA Efficiency Analysis across {len(pairs)} pairs")
        self.logger.info(f"   Min ER threshold: {min_efficiency_ratio}")
        
        results = {
            'timestamp': datetime.now(),
            'timeframe': timeframe,
            'min_er_threshold': min_efficiency_ratio,
            'trending_pairs': [],
            'ranging_pairs': [],
            'analysis_summary': {}
        }
        
        try:
            tf_minutes = self._timeframe_to_minutes(timeframe)
            
            for pair in pairs:
                try:
                    df = self.data_fetcher.fetch_latest_data(pair, tf_minutes, 200)
                    
                    if df is None or len(df) < 50:
                        self.logger.warning(f"⚠️ Insufficient data for {pair}")
                        continue
                    
                    # Get latest KAMA efficiency ratio
                    if 'kama_10_er' in df.columns:
                        latest_er = df['kama_10_er'].iloc[-1]
                        avg_er = df['kama_10_er'].tail(20).mean()
                        
                        pair_analysis = {
                            'pair': pair,
                            'latest_er': latest_er,
                            'avg_er_20': avg_er,
                            'regime': 'trending' if latest_er >= min_efficiency_ratio else 'ranging',
                            'er_trend': 'increasing' if latest_er > avg_er else 'decreasing'
                        }
                        
                        if latest_er >= min_efficiency_ratio:
                            results['trending_pairs'].append(pair_analysis)
                        else:
                            results['ranging_pairs'].append(pair_analysis)
                        
                        self.logger.info(f"   {pair}: ER={latest_er:.3f} ({pair_analysis['regime']})")
                    else:
                        self.logger.warning(f"⚠️ No KAMA data available for {pair}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error analyzing {pair}: {e}")
                    continue
            
            # Summary statistics
            total_analyzed = len(results['trending_pairs']) + len(results['ranging_pairs'])
            trending_count = len(results['trending_pairs'])
            ranging_count = len(results['ranging_pairs'])
            
            results['analysis_summary'] = {
                'total_pairs_analyzed': total_analyzed,
                'trending_pairs_count': trending_count,
                'ranging_pairs_count': ranging_count,
                'trending_percentage': (trending_count / total_analyzed * 100) if total_analyzed > 0 else 0,
                'market_regime': 'mostly_trending' if trending_count > ranging_count else 'mostly_ranging'
            }
            
            self.logger.info(f"📊 Market Regime Summary:")
            self.logger.info(f"   Trending pairs: {trending_count}/{total_analyzed} ({results['analysis_summary']['trending_percentage']:.1f}%)")
            self.logger.info(f"   Market regime: {results['analysis_summary']['market_regime']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in efficiency analysis: {e}")
            return results
    
    def scan_multi_strategy_dashboard(
        self,
        pair: str,
        timeframe: str = '5m'
    ) -> Dict:
        """
        Create a comprehensive dashboard view for a single pair across all strategies
        
        Args:
            pair: Trading pair to analyze
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with comprehensive strategy analysis
        """
        self.logger.info(f"📊 Multi-Strategy Dashboard for {pair}")
        
        dashboard = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'market_overview': {},
            'strategy_signals': {},
            'confluence_analysis': {},
            'risk_assessment': {},
            'recommendations': {}
        }
        
        try:
            # Get enhanced data
            tf_minutes = self._timeframe_to_minutes(timeframe)
            df = self.data_fetcher.fetch_latest_data(pair, tf_minutes, 300)
            
            if df is None or len(df) < 100:
                self.logger.error(f"❌ Insufficient data for {pair}")
                return dashboard
            
            latest = df.iloc[-1]
            
            # Market Overview
            dashboard['market_overview'] = {
                'current_price': latest['close'],
                'price_change_24h': 0,  # Calculate if needed
                'volatility_bb_width': latest.get('bb_width', 0),
                'volume_ratio': latest.get('volume_ratio', 1.0),
                'kama_efficiency_ratio': latest.get('kama_10_er', 0),
                'market_regime': 'trending' if latest.get('kama_10_er', 0) > 0.5 else 'ranging'
            }
            
            # Strategy Signals
            strategies_to_test = ['ema', 'combined']
            if hasattr(config, 'MACD_EMA_STRATEGY') and config.MACD_EMA_STRATEGY:
                strategies_to_test.append('macd')
            if hasattr(config, 'KAMA_STRATEGY') and config.KAMA_STRATEGY:
                strategies_to_test.append('kama')
            
            for strategy_name in strategies_to_test:
                try:
                    signal = self.signal_detector.detect_strategy_signal(
                        strategy_name, df, pair, getattr(config, 'SPREAD_PIPS', 1.5), timeframe
                    )
                    
                    dashboard['strategy_signals'][strategy_name] = {
                        'has_signal': signal is not None,
                        'signal_type': signal['signal_type'] if signal else None,
                        'confidence': signal['confidence_score'] if signal else 0,
                        'trigger_reason': signal.get('trigger_reason', '') if signal else '',
                        'strategy_specific_data': self._extract_strategy_data(signal) if signal else {}
                    }
                    
                except Exception as e:
                    dashboard['strategy_signals'][strategy_name] = {
                        'error': str(e),
                        'has_signal': False
                    }
            
            # Confluence Analysis
            confluence = self.signal_detector.analyze_signal_confluence(
                df, pair, getattr(config, 'SPREAD_PIPS', 1.5), timeframe
            )
            
            dashboard['confluence_analysis'] = {
                'confluence_score': confluence.get('confluence_score', 0),
                'dominant_direction': confluence.get('dominant_direction', 'NEUTRAL'),
                'confidence_weighted_direction': confluence.get('confidence_weighted_direction', 'NEUTRAL'),
                'bull_strategies': len(confluence.get('bull_signals', [])),
                'bear_strategies': len(confluence.get('bear_signals', [])),
                'agreement_level': 'high' if confluence.get('confluence_score', 0) > 0.7 else 'medium' if confluence.get('confluence_score', 0) > 0.4 else 'low'
            }
            
            # Display dashboard
            self._display_strategy_dashboard(dashboard)
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"❌ Error creating dashboard for {pair}: {e}")
            return dashboard
    
    def _scan_single_pair(
        self,
        pair: str,
        timeframe: str,
        strategy: str,
        enable_claude_analysis: bool
    ) -> List[Dict]:
        """Scan a single trading pair for signals"""
        try:
            # Convert timeframe and fetch data
            tf_minutes = self._timeframe_to_minutes(timeframe)
            df = self.data_fetcher.fetch_latest_data(pair, tf_minutes, 300)
            
            min_bars = getattr(config, 'MIN_BARS_FOR_ANALYSIS', 100)
            if df is None or len(df) < min_bars:
                return []
            
            # Detect signals based on strategy
            signals = []
            spread_pips = getattr(config, 'SPREAD_PIPS', 1.5)
            
            if strategy == 'all':
                # Get signals from all active strategies
                all_signals = self.signal_detector.detect_signals(
                    df, pair, spread_pips, timeframe
                )
                signals.extend(all_signals)
            
            elif strategy in ['combined', 'ema', 'macd', 'kama'] or strategy.startswith('scalping_'):
                # Get signal from specific strategy
                signal = self.signal_detector.detect_strategy_signal(
                    strategy, df, pair, spread_pips, timeframe
                )
                if signal:
                    signals.append(signal)
            
            else:
                self.logger.warning(f"⚠️ Unknown strategy: {strategy}")
                return []
            
            # Filter signals by confidence
            min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)
            filtered_signals = [s for s in signals if s['confidence_score'] >= min_confidence]
            
            # Add Claude analysis if enabled and we have high-confidence signals
            if enable_claude_analysis and self.claude_api and filtered_signals:
                for signal in filtered_signals:
                    if signal['confidence_score'] >= 0.7:  # Only analyze high-confidence signals
                        try:
                            claude_analysis = self._get_claude_analysis(signal, df, pair)
                            signal['claude_analysis'] = claude_analysis
                        except Exception as e:
                            self.logger.error(f"❌ Claude analysis failed for {pair}: {e}")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {pair}: {e}")
            return []
    
    def _get_confluence_analysis(self, pair: str, timeframe: str) -> Optional[Dict]:
        """Get confluence analysis for a trading pair"""
        try:
            tf_minutes = self._timeframe_to_minutes(timeframe)
            df = self.data_fetcher.fetch_latest_data(pair, tf_minutes, 300)
            
            if df is None or len(df) < 100:
                return None
            
            confluence = self.signal_detector.analyze_signal_confluence(
                df, pair, getattr(config, 'SPREAD_PIPS', 1.5), timeframe
            )
            
            return confluence
            
        except Exception as e:
            self.logger.error(f"❌ Error getting confluence for {pair}: {e}")
            return None
    
    def _get_claude_analysis(self, signal: Dict, df: pd.DataFrame, pair: str) -> Optional[str]:
        """Get Claude AI analysis for a signal"""
        try:
            if not self.claude_api:
                return "Claude analysis not available"
            
            # Prepare data for Claude
            latest_data = df.tail(5)
            
            analysis_prompt = self._build_claude_prompt(signal, latest_data, pair)
            
            claude_response = self.claude_api.analyze_signal(
                signal_data=signal,
                market_data=latest_data.to_dict('records'),
                custom_prompt=analysis_prompt
            )
            
            return claude_response
            
        except Exception as e:
            self.logger.error(f"❌ Claude analysis error: {e}")
            return None
    
    def _build_claude_prompt(self, signal: Dict, latest_data: pd.DataFrame, pair: str) -> str:
        """Build Claude analysis prompt with KAMA context"""
        strategy = signal.get('detector_strategy', 'unknown')
        
        prompt = f"""
        Analyze this {strategy.upper()} trading signal for {pair}:
        
        Signal Details:
        - Type: {signal['signal_type']}
        - Confidence: {signal['confidence_score']:.1%}
        - Price: {signal['signal_price']:.5f}
        - Strategy: {strategy}
        - Trigger: {signal.get('trigger_reason', 'N/A')}
        """
        
        # Add strategy-specific details
        if 'kama' in strategy.lower():
            prompt += f"""
        
        KAMA-Specific Data:
        - KAMA Value: {signal.get('kama_value', 'N/A')}
        - Efficiency Ratio: {signal.get('efficiency_ratio', 'N/A')}
        - KAMA Trend: {signal.get('kama_trend', 'N/A')}
        - Market Regime: {'Trending' if signal.get('efficiency_ratio', 0) > 0.5 else 'Ranging'}
        """
        
        prompt += """
        
        Please provide:
        1. Signal quality assessment (1-10 scale)
        2. Key strengths and concerns
        3. Risk/reward analysis
        4. Entry and exit recommendations
        5. Overall verdict (TAKE/PASS/MONITOR)
        
        Keep analysis concise and actionable.
        """
        
        return prompt
    
    def _extract_strategy_data(self, signal: Dict) -> Dict:
        """Extract strategy-specific data from signal"""
        strategy_data = {}
        
        strategy_name = signal.get('detector_strategy', '')
        
        if 'kama' in strategy_name:
            strategy_data = {
                'kama_value': signal.get('kama_value'),
                'efficiency_ratio': signal.get('efficiency_ratio'),
                'kama_trend': signal.get('kama_trend'),
                'kama_slope': signal.get('kama_slope')
            }
        
        elif 'ema' in strategy_name:
            strategy_data = {
                'ema_9': signal.get('ema_9'),
                'ema_21': signal.get('ema_21'),
                'ema_200': signal.get('ema_200')
            }
        
        elif 'combined' in strategy_name:
            strategy_data = {
                'combination_type': signal.get('combination_type'),
                'source_strategies': signal.get('source_strategies', []),
                'num_agreeing_strategies': signal.get('num_agreeing_strategies')
            }
        
        return strategy_data
    
    def _display_strategy_dashboard(self, dashboard: Dict):
        """Display the strategy dashboard in a formatted way"""
        pair = dashboard['pair']
        timestamp = dashboard['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"📊 === STRATEGY DASHBOARD: {pair} ({timestamp}) ===")
        
        # Market Overview
        overview = dashboard['market_overview']
        self.logger.info(f"🏛️ Market Overview:")
        self.logger.info(f"   Price: {overview.get('current_price', 0):.5f}")
        self.logger.info(f"   Market Regime: {overview.get('market_regime', 'unknown').upper()}")
        self.logger.info(f"   KAMA ER: {overview.get('kama_efficiency_ratio', 0):.3f}")
        
        # Strategy Signals
        self.logger.info(f"🎯 Strategy Signals:")
        signals = dashboard['strategy_signals']
        for strategy, signal_data in signals.items():
            if signal_data.get('has_signal'):
                signal_type = signal_data['signal_type']
                confidence = signal_data['confidence']
                emoji = '🟢' if signal_type == 'BULL' else '🔴'
                self.logger.info(f"   {emoji} {strategy.upper()}: {signal_type} ({confidence:.1%})")
            else:
                self.logger.info(f"   ➖ {strategy.upper()}: No signal")
        
        # Confluence Analysis
        confluence = dashboard['confluence_analysis']
        self.logger.info(f"🤝 Confluence Analysis:")
        self.logger.info(f"   Direction: {confluence.get('dominant_direction', 'NEUTRAL')}")
        self.logger.info(f"   Agreement: {confluence.get('agreement_level', 'unknown').upper()}")
        self.logger.info(f"   Score: {confluence.get('confluence_score', 0):.2f}")
        
        self.logger.info(f"📊 === END DASHBOARD ===")
    
    def _update_scan_stats(self, signals: List[Dict], strategy: str):
        """Update scanning statistics"""
        self.scan_stats['total_scans'] += 1
        self.scan_stats['total_signals'] += len(signals)
        self.scan_stats['last_scan_time'] = datetime.now()
        
        # Track by strategy
        for signal in signals:
            strategy_name = signal.get('detector_strategy', strategy)
            if strategy_name not in self.scan_stats['strategy_signals']:
                self.scan_stats['strategy_signals'][strategy_name] = 0
            self.scan_stats['strategy_signals'][strategy_name] += 1
    
    def _display_scan_summary(self, signals: List[Dict], scan_duration: float, strategy: str):
        """Display scan summary"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if signals:
            self.logger.info(f"🎯 [{timestamp}] Found {len(signals)} signal(s) in {scan_duration:.1f}s")
            
            for signal in signals:
                strategy_name = signal.get('detector_strategy', strategy)
                signal_type = signal['signal_type']
                confidence = signal['confidence_score']
                pair = signal['epic']
                
                # Strategy-specific emoji
                strategy_emoji = {
                    'ema': '📈',
                    'macd': '📊', 
                    'kama': '🔄',
                    'combined': '🤝',
                    'scalping_ultra_fast': '⚡',
                    'scalping_aggressive': '🏃',
                    'scalping_conservative': '🚶'
                }.get(strategy_name, '📋')
                
                type_emoji = '🟢' if signal_type == 'BULL' else '🔴'
                
                self.logger.info(f"   {strategy_emoji} {strategy_name.upper()}: {type_emoji} {signal_type} "
                               f"{pair} ({confidence:.1%})")
                
                # Show additional details for specific strategies
                if 'kama' in strategy_name:
                    er = signal.get('efficiency_ratio')
                    if er:
                        regime = 'Trending' if er > 0.5 else 'Ranging'
                        self.logger.info(f"      ER: {er:.3f} ({regime})")
                
                elif 'combined' in strategy_name:
                    sources = signal.get('source_strategies', [])
                    if sources:
                        self.logger.info(f"      Sources: {', '.join(sources)}")
        else:
            self.logger.info(f"📊 [{timestamp}] No signals found in {scan_duration:.1f}s")
    
    def _display_scan_results(self, signals: List[Dict], confluence_results: List[Dict], strategy: str):
        """Display batch scan results"""
        self.logger.info(f"📊 Scan Results Summary ({strategy} strategy)")
        self.logger.info(f"   Total signals found: {len(signals)}")
        
        if signals:
            # Group by strategy
            strategy_groups = {}
            for signal in signals:
                strat_name = signal.get('detector_strategy', strategy)
                if strat_name not in strategy_groups:
                    strategy_groups[strat_name] = []
                strategy_groups[strat_name].append(signal)
            
            for strat_name, strat_signals in strategy_groups.items():
                avg_confidence = sum(s['confidence_score'] for s in strat_signals) / len(strat_signals)
                bull_count = sum(1 for s in strat_signals if s['signal_type'] == 'BULL')
                bear_count = len(strat_signals) - bull_count
                
                self.logger.info(f"   {strat_name.upper()}: {len(strat_signals)} signals "
                               f"(🟢{bull_count} 🔴{bear_count}) Avg: {avg_confidence:.1%}")
        
        # Display confluence results
        if confluence_results:
            self.logger.info(f"🤝 Confluence Analysis:")
            strong_confluence = [c for c in confluence_results if c['confluence_score'] > 0.6]
            
            if strong_confluence:
                for confluence in strong_confluence:
                    pair = confluence['epic']
                    score = confluence['confluence_score']
                    direction = confluence['dominant_direction']
                    
                    self.logger.info(f"   {pair}: {direction} (Score: {score:.2f})")
    
    def _send_signal_notifications(self, signals: List[Dict]):
        """Send notifications for new signals"""
        try:
            for signal in signals:
                # Only notify for high-confidence signals
                if signal['confidence_score'] >= 0.7:
                    self.notification_manager.send_signal_alert(signal)
        
        except Exception as e:
            self.logger.error(f"❌ Error sending notifications: {e}")
    
    def _display_final_stats(self):
        """Display final scanning statistics"""
        self.logger.info("📊 Final Scanning Statistics:")
        self.logger.info(f"   Total scans completed: {self.scan_stats['total_scans']}")
        self.logger.info(f"   Total signals detected: {self.scan_stats['total_signals']}")
        
        if self.scan_stats['total_scans'] > 0:
            avg_signals = self.scan_stats['total_signals'] / self.scan_stats['total_scans']
            self.logger.info(f"   Average signals per scan: {avg_signals:.1f}")
        
        if self.scan_stats['strategy_signals']:
            self.logger.info("   Signals by strategy:")
            for strategy, count in self.scan_stats['strategy_signals'].items():
                percentage = (count / self.scan_stats['total_signals']) * 100
                self.logger.info(f"     {strategy}: {count} ({percentage:.1f}%)")
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return mapping.get(timeframe, 5)