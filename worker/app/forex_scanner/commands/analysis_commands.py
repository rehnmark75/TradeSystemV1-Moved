# commands/analysis_commands.py
"""
Analysis Commands Module
Handles specialized analysis operations: BB testing, EMA config listing, technical analysis
"""

import logging
from typing import List, Dict, Optional

try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    from analysis.technical import TechnicalAnalyzer
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner.analysis.technical import TechnicalAnalyzer
    from forex_scanner import config


class AnalysisCommands:
    """Analysis command implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def test_bb_data(self, epic: str) -> bool:
        """Test Bollinger Band data availability and calculation"""
        self.logger.info(f"📊 Testing Bollinger Band data for {epic}")
        
        try:
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            technical_analyzer = TechnicalAnalyzer()
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get enhanced data
            df = detector.data_fetcher.get_enhanced_data(epic, pair, timeframe='5m', lookback_hours=48)
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for BB analysis")
                return False
            
            self.logger.info(f"✅ Data available: {len(df)} bars")
            
            # Check if BB indicators exist
            bb_columns = ['bb_upper', 'bb_lower', 'bb_middle', 'bb_position']
            existing_bb = [col for col in bb_columns if col in df.columns]
            missing_bb = [col for col in bb_columns if col not in df.columns]
            
            self.logger.info(f"📈 BB Indicators Status:")
            if existing_bb:
                self.logger.info(f"   ✅ Existing: {existing_bb}")
            if missing_bb:
                self.logger.info(f"   ❌ Missing: {missing_bb}")
                
                # Add missing BB indicators
                self.logger.info("🔄 Adding missing BB indicators...")
                df = technical_analyzer.add_bollinger_bands(df)
                self.logger.info("✅ BB indicators added")
            
            # Analyze BB data quality
            latest = df.iloc[-1]
            
            self.logger.info(f"📊 Latest BB Data:")
            self.logger.info(f"   Current Price: {latest['close']:.5f}")
            self.logger.info(f"   BB Upper: {latest.get('bb_upper', 'N/A'):.5f}")
            self.logger.info(f"   BB Middle: {latest.get('bb_middle', 'N/A'):.5f}")
            self.logger.info(f"   BB Lower: {latest.get('bb_lower', 'N/A'):.5f}")
            self.logger.info(f"   BB Position: {latest.get('bb_position', 'N/A'):.2f}")
            self.logger.info(f"   BB Width: {latest.get('bb_width', 'N/A'):.5f}")
            
            # Test BB filter conditions
            self._test_bb_filter_conditions(df)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ BB data test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compare_bb_filters(self, epic: str = None, days: int = 7) -> bool:
        """Compare signal performance with and without BB filters"""
        test_epic = epic or 'CS.D.EURUSD.CEEM.IP'
        self.logger.info(f"🔬 Comparing BB filter performance for {test_epic}")
        
        try:
            # Store original BB settings
            original_bb_filter = getattr(config, 'ENABLE_BB_FILTER', False)
            original_bb_extremes = getattr(config, 'ENABLE_BB_EXTREMES_FILTER', False)
            
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            comparison_results = {}
            
            # Test scenarios
            test_scenarios = [
                ('no_bb_filters', {'ENABLE_BB_FILTER': False, 'ENABLE_BB_EXTREMES_FILTER': False}),
                ('bb_midline_only', {'ENABLE_BB_FILTER': True, 'ENABLE_BB_EXTREMES_FILTER': False}),
                ('bb_extremes_only', {'ENABLE_BB_FILTER': False, 'ENABLE_BB_EXTREMES_FILTER': True}),
                ('both_bb_filters', {'ENABLE_BB_FILTER': True, 'ENABLE_BB_EXTREMES_FILTER': True})
            ]
            
            for scenario_name, bb_settings in test_scenarios:
                self.logger.info(f"\n🧪 Testing scenario: {scenario_name}")
                
                # Apply BB settings
                for setting, value in bb_settings.items():
                    setattr(config, setting, value)
                    self.logger.info(f"   {setting}: {value}")
                
                # Run backtest
                results = detector.backtest_signals(
                    epic_list=[test_epic],
                    lookback_days=days,
                    use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                    spread_pips=config.SPREAD_PIPS,
                    timeframe='5m'
                )
                
                # Analyze results
                from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
                performance_analyzer = PerformanceAnalyzer()
                performance = performance_analyzer.analyze_performance(results)
                
                comparison_results[scenario_name] = {
                    'signals': results,
                    'performance': performance,
                    'settings': bb_settings
                }
                
                signal_count = performance.get('total_signals', 0)
                avg_confidence = performance.get('average_confidence', 0)
                self.logger.info(f"   Results: {signal_count} signals, {avg_confidence:.1%} avg confidence")
            
            # Display comparison
            self._display_bb_comparison(comparison_results)
            
            # Restore original settings
            config.ENABLE_BB_FILTER = original_bb_filter
            config.ENABLE_BB_EXTREMES_FILTER = original_bb_extremes
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ BB filter comparison failed: {e}")
            return False
    
    def list_ema_configs(self) -> bool:
        """List all available EMA configurations"""
        self.logger.info("📋 Available EMA Configurations:")
        
        try:
            if not hasattr(config, 'EMA_STRATEGY_CONFIG'):
                self.logger.error("❌ EMA_STRATEGY_CONFIG not found in config")
                return False
            
            active_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
            
            self.logger.info("=" * 70)
            header = f"{'Config':<15} {'Short':<6} {'Long':<6} {'Trend':<6} {'Status':<8} {'Description'}"
            self.logger.info(header)
            self.logger.info("-" * 70)
            
            for config_name, settings in config.EMA_STRATEGY_CONFIG.items():
                status = "ACTIVE" if config_name == active_config else ""
                description = self._get_ema_config_description(config_name, settings)
                
                row = (f"{config_name:<15} "
                       f"{settings['short']:<6} "
                       f"{settings['long']:<6} "
                       f"{settings['trend']:<6} "
                       f"{status:<8} "
                       f"{description}")
                
                self.logger.info(row)
            
            self.logger.info("=" * 70)
            self.logger.info(f"💡 Current active config: {active_config}")
            self.logger.info(f"💡 Change with: --ema-config <config_name>")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list EMA configs: {e}")
            return False
    
    def analyze_market_conditions(self, epic: str = None, hours: int = 24) -> bool:
        """Analyze current market conditions for trading"""
        test_epic = epic or 'CS.D.EURUSD.CEEM.IP'
        self.logger.info(f"🌍 Analyzing market conditions for {test_epic}")
        
        try:
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(test_epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get recent data
            df = detector.data_fetcher.get_enhanced_data(
                test_epic, pair, timeframe='5m', lookback_hours=hours
            )
            
            if df is None or len(df) < 50:
                self.logger.error("❌ Insufficient data for market analysis")
                return False
            
            # Analyze market conditions
            self._analyze_volatility(df)
            self._analyze_trend_strength(df)
            self._analyze_volume_patterns(df)
            self._analyze_trading_session()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market conditions analysis failed: {e}")
            return False
    
    def test_indicator_calculations(self, epic: str = None) -> bool:
        """Test various technical indicator calculations"""
        test_epic = epic or 'CS.D.EURUSD.CEEM.IP'
        self.logger.info(f"🧮 Testing indicator calculations for {test_epic}")
        
        try:
            # Initialize components
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            technical_analyzer = TechnicalAnalyzer()
            
            # Get pair info
            pair_info = config.PAIR_INFO.get(test_epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            # Get data
            df = detector.data_fetcher.get_enhanced_data(test_epic, pair, timeframe='5m', lookback_hours=168)
            
            if df is None or len(df) < 200:
                self.logger.error("❌ Insufficient data for indicator testing")
                return False
            
            self.logger.info(f"📊 Testing indicators on {len(df)} bars")
            
            # Test EMA calculations
            self._test_ema_indicators(df, technical_analyzer)
            
            # Test MACD calculations
            self._test_macd_indicators(df, technical_analyzer)
            
            # Test Bollinger Bands
            self._test_bb_indicators(df, technical_analyzer)
            
            # Test Support/Resistance
            self._test_sr_indicators(df, technical_analyzer, pair)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Indicator testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_bb_filter_conditions(self, df):
        """Test BB filter conditions on data"""
        if len(df) < 20:
            return
        
        recent_data = df.tail(20)
        
        # Test midline filter
        if 'bb_middle' in recent_data.columns:
            above_midline = (recent_data['close'] > recent_data['bb_middle']).sum()
            below_midline = (recent_data['close'] < recent_data['bb_middle']).sum()
            
            self.logger.info(f"🎯 BB Midline Analysis (last 20 bars):")
            self.logger.info(f"   Above midline: {above_midline} bars")
            self.logger.info(f"   Below midline: {below_midline} bars")
        
        # Test extremes filter
        if 'bb_position' in recent_data.columns:
            extreme_upper = (recent_data['bb_position'] > 0.8).sum()
            extreme_lower = (recent_data['bb_position'] < 0.2).sum()
            middle_zone = ((recent_data['bb_position'] >= 0.2) & (recent_data['bb_position'] <= 0.8)).sum()
            
            self.logger.info(f"🎯 BB Extremes Analysis:")
            self.logger.info(f"   Upper extreme (>80%): {extreme_upper} bars")
            self.logger.info(f"   Lower extreme (<20%): {extreme_lower} bars")
            self.logger.info(f"   Middle zone (20-80%): {middle_zone} bars")
    
    def _display_bb_comparison(self, comparison_results: Dict):
        """Display BB filter comparison results"""
        self.logger.info(f"\n📊 BOLLINGER BAND FILTER COMPARISON:")
        self.logger.info("=" * 80)
        
        # Header
        header = f"{'Scenario':<18} {'Signals':<8} {'Avg Conf':<9} {'Win Rate':<9} {'Quality':<8}"
        self.logger.info(header)
        self.logger.info("-" * 80)
        
        for scenario_name, data in comparison_results.items():
            performance = data['performance']
            
            signal_count = performance.get('total_signals', 0)
            avg_confidence = performance.get('average_confidence', 0)
            win_rate = performance.get('win_rate', 0)
            
            # Calculate quality score (combination of factors)
            quality_score = (avg_confidence * 0.4) + (win_rate * 0.6)
            
            row = (f"{scenario_name:<18} "
                   f"{signal_count:<8} "
                   f"{avg_confidence:<9.1%} "
                   f"{win_rate:<9.1%} "
                   f"{quality_score:<8.1%}")
            
            self.logger.info(row)
        
        self.logger.info("=" * 80)
        
        # Recommendations
        best_quality = max(comparison_results.items(), 
                          key=lambda x: (x[1]['performance'].get('average_confidence', 0) * 0.4 + 
                                       x[1]['performance'].get('win_rate', 0) * 0.6))
        
        self.logger.info(f"🏆 Best quality: {best_quality[0]}")
    
    def _get_ema_config_description(self, config_name: str, settings: Dict) -> str:
        """Get description for EMA configuration"""
        descriptions = {
            'default': 'Standard 9/21/200 EMAs - balanced approach',
            'aggressive': 'Fast 5/13/100 EMAs - quick signals, more noise',
            'conservative': 'Slow 12/26/200 EMAs - fewer but stronger signals',
            'scalping': 'Ultra-fast 3/8/50 EMAs - for scalping strategies',
            'swing': 'Long-term 21/50/200 EMAs - for swing trading'
        }
        
        return descriptions.get(config_name, 'Custom EMA configuration')
    
    def _analyze_volatility(self, df):
        """Analyze market volatility"""
        if len(df) < 20:
            return
        
        recent_data = df.tail(20)
        
        # Calculate recent volatility metrics
        price_ranges = recent_data['high'] - recent_data['low']
        avg_range_pips = (price_ranges.mean() * 10000)
        max_range_pips = (price_ranges.max() * 10000)
        
        # Price change volatility
        price_changes = recent_data['close'].pct_change().dropna()
        volatility = price_changes.std() * 100
        
        self.logger.info(f"📈 VOLATILITY ANALYSIS (last 20 bars):")
        self.logger.info(f"   Average range: {avg_range_pips:.1f} pips")
        self.logger.info(f"   Maximum range: {max_range_pips:.1f} pips")
        self.logger.info(f"   Price volatility: {volatility:.2f}%")
        
        # Volatility assessment
        if avg_range_pips < 5:
            self.logger.info(f"   📊 Assessment: Low volatility - market is quiet")
        elif avg_range_pips > 20:
            self.logger.info(f"   📊 Assessment: High volatility - market is active")
        else:
            self.logger.info(f"   📊 Assessment: Normal volatility - good for trading")
    
    def _analyze_trend_strength(self, df):
        """Analyze trend strength using EMAs"""
        if len(df) < 200 or 'ema_9' not in df.columns:
            return
        
        latest = df.iloc[-1]
        
        # EMA alignment
        ema_9 = latest.get('ema_9', 0)
        ema_21 = latest.get('ema_21', 0)
        ema_200 = latest.get('ema_200', 0)
        current_price = latest['close']
        
        self.logger.info(f"📊 TREND STRENGTH ANALYSIS:")
        self.logger.info(f"   Current Price: {current_price:.5f}")
        self.logger.info(f"   EMA 9: {ema_9:.5f}")
        self.logger.info(f"   EMA 21: {ema_21:.5f}")
        self.logger.info(f"   EMA 200: {ema_200:.5f}")
        
        # Trend assessment
        if current_price > ema_9 > ema_21 > ema_200:
            trend_strength = "Strong Bullish"
        elif current_price < ema_9 < ema_21 < ema_200:
            trend_strength = "Strong Bearish"
        elif current_price > ema_200 and ema_9 > ema_200:
            trend_strength = "Weak Bullish"
        elif current_price < ema_200 and ema_9 < ema_200:
            trend_strength = "Weak Bearish"
        else:
            trend_strength = "Ranging/Mixed"
        
        self.logger.info(f"   📈 Trend Assessment: {trend_strength}")
        
        # EMA separation (strength indicator)
        ema_separation_pips = abs(ema_9 - ema_21) * 10000
        self.logger.info(f"   📏 EMA 9-21 separation: {ema_separation_pips:.1f} pips")
        
        if ema_separation_pips < 2:
            self.logger.info(f"   💡 Tip: EMAs are very close - expect choppy conditions")
        elif ema_separation_pips > 10:
            self.logger.info(f"   💡 Tip: EMAs are well separated - strong trend conditions")
    
    def _analyze_volume_patterns(self, df):
        """Analyze volume patterns"""
        if 'ltv' not in df.columns or len(df) < 20:
            self.logger.info(f"📊 VOLUME ANALYSIS: No volume data available")
            return
        
        recent_data = df.tail(20)
        
        # Volume metrics
        avg_volume = recent_data['ltv'].mean()
        current_volume = recent_data['ltv'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        volume_trend = recent_data['ltv'].tail(5).mean() / recent_data['ltv'].head(5).mean()
        
        self.logger.info(f"📊 VOLUME ANALYSIS:")
        self.logger.info(f"   Current volume: {current_volume:,.0f}")
        self.logger.info(f"   Average volume (20 bars): {avg_volume:,.0f}")
        self.logger.info(f"   Volume ratio: {volume_ratio:.2f}x")
        self.logger.info(f"   Volume trend: {volume_trend:.2f}x")
        
        # Volume assessment
        if volume_ratio > 1.5:
            self.logger.info(f"   📈 Assessment: High volume - strong market interest")
        elif volume_ratio < 0.5:
            self.logger.info(f"   📉 Assessment: Low volume - weak market interest")
        else:
            self.logger.info(f"   📊 Assessment: Normal volume levels")
    
    def _analyze_trading_session(self):
        """Analyze current trading session"""
        from datetime import datetime
        import pytz
        
        now = datetime.now(pytz.timezone(config.USER_TIMEZONE))
        hour = now.hour
        
        self.logger.info(f"🕐 TRADING SESSION ANALYSIS:")
        self.logger.info(f"   Current time: {now.strftime('%H:%M %Z')}")
        
        # Session identification
        if 8 <= hour < 13:
            session = "London"
            assessment = "High liquidity, good for trend following"
        elif 13 <= hour < 17:
            session = "London-NY Overlap"
            assessment = "Highest liquidity, best for all strategies"
        elif 17 <= hour < 22:
            session = "New York"
            assessment = "Good liquidity, trend continuation"
        elif 22 <= hour or hour < 2:
            session = "Sydney"
            assessment = "Lower liquidity, range-bound often"
        else:
            session = "Asian"
            assessment = "Moderate liquidity, often ranging"
        
        self.logger.info(f"   Current session: {session}")
        self.logger.info(f"   📊 Assessment: {assessment}")
        
        # Weekend check
        if now.weekday() >= 5:  # Saturday or Sunday
            self.logger.info(f"   ⚠️ Weekend: Markets closed, limited activity expected")
    
    def _test_ema_indicators(self, df, technical_analyzer):
        """Test EMA indicator calculations"""
        self.logger.info(f"\n📈 Testing EMA Indicators:")
        
        # Test different EMA periods
        test_periods = [9, 21, 50, 200]
        
        for period in test_periods:
            col_name = f'ema_{period}'
            if col_name in df.columns:
                latest_value = df[col_name].iloc[-1]
                # Test for NaN values
                nan_count = df[col_name].isna().sum()
                self.logger.info(f"   ✅ EMA {period}: {latest_value:.5f} (NaN: {nan_count})")
            else:
                self.logger.info(f"   ❌ EMA {period}: Missing")
    
    def _test_macd_indicators(self, df, technical_analyzer):
        """Test MACD indicator calculations"""
        self.logger.info(f"\n📊 Testing MACD Indicators:")
        
        macd_columns = ['macd_line', 'macd_signal', 'macd_histogram']
        
        # Add MACD if missing
        if not all(col in df.columns for col in macd_columns):
            self.logger.info(f"   🔄 Adding MACD indicators...")
            df = technical_analyzer.add_macd_indicators(df)
        
        for col in macd_columns:
            if col in df.columns:
                latest_value = df[col].iloc[-1]
                nan_count = df[col].isna().sum()
                self.logger.info(f"   ✅ {col}: {latest_value:.6f} (NaN: {nan_count})")
            else:
                self.logger.info(f"   ❌ {col}: Missing")
    
    def _test_bb_indicators(self, df, technical_analyzer):
        """Test Bollinger Band indicators"""
        self.logger.info(f"\n📊 Testing Bollinger Band Indicators:")
        
        bb_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width']
        
        # Add BB if missing
        if not all(col in df.columns for col in bb_columns[:3]):
            self.logger.info(f"   🔄 Adding Bollinger Band indicators...")
            df = technical_analyzer.add_bollinger_bands(df)
        
        for col in bb_columns:
            if col in df.columns:
                latest_value = df[col].iloc[-1]
                nan_count = df[col].isna().sum()
                self.logger.info(f"   ✅ {col}: {latest_value:.5f} (NaN: {nan_count})")
            else:
                self.logger.info(f"   ❌ {col}: Missing")
    
    def _test_sr_indicators(self, df, technical_analyzer, pair):
        """Test Support/Resistance indicators"""
        self.logger.info(f"\n📊 Testing Support/Resistance Indicators:")
        
        sr_columns = ['nearest_support', 'nearest_resistance', 'distance_to_support_pips', 'distance_to_resistance_pips']
        
        # Add S/R if missing
        if not all(col in df.columns for col in sr_columns):
            self.logger.info(f"   🔄 Adding Support/Resistance indicators...")
            df = technical_analyzer.add_support_resistance_to_df(df, pair)
        
        for col in sr_columns:
            if col in df.columns:
                latest_value = df[col].iloc[-1]
                if pd.isna(latest_value):
                    self.logger.info(f"   ⚠️ {col}: N/A")
                else:
                    self.logger.info(f"   ✅ {col}: {latest_value:.5f}")
            else:
                self.logger.info(f"   ❌ {col}: Missing")
    
    def benchmark_indicator_performance(self, epic: str = None, days: int = 30) -> bool:
        """Benchmark different indicator combinations"""
        test_epic = epic or 'CS.D.EURUSD.CEEM.IP'
        self.logger.info(f"⚡ Benchmarking indicator performance for {test_epic}")
        
        try:
            # Store original settings
            original_bb_filter = getattr(config, 'ENABLE_BB_FILTER', False)
            original_volume_req = getattr(config, 'REQUIRE_VOLUME_CONFIRMATION', False)
            original_ema_sep = getattr(config, 'REQUIRE_EMA_SEPARATION', False)
            
            db_manager = DatabaseManager(config.DATABASE_URL)
            detector = SignalDetector(db_manager, config.USER_TIMEZONE)
            
            # Test different indicator combinations
            test_combinations = [
                ('baseline', {'ENABLE_BB_FILTER': False, 'REQUIRE_VOLUME_CONFIRMATION': False, 'REQUIRE_EMA_SEPARATION': False}),
                ('with_bb', {'ENABLE_BB_FILTER': True, 'REQUIRE_VOLUME_CONFIRMATION': False, 'REQUIRE_EMA_SEPARATION': False}),
                ('with_volume', {'ENABLE_BB_FILTER': False, 'REQUIRE_VOLUME_CONFIRMATION': True, 'REQUIRE_EMA_SEPARATION': False}),
                ('with_ema_sep', {'ENABLE_BB_FILTER': False, 'REQUIRE_VOLUME_CONFIRMATION': False, 'REQUIRE_EMA_SEPARATION': True}),
                ('all_filters', {'ENABLE_BB_FILTER': True, 'REQUIRE_VOLUME_CONFIRMATION': True, 'REQUIRE_EMA_SEPARATION': True})
            ]
            
            benchmark_results = {}
            
            for combo_name, settings in test_combinations:
                self.logger.info(f"\n🧪 Testing: {combo_name}")
                
                # Apply settings
                for setting, value in settings.items():
                    setattr(config, setting, value)
                
                # Run backtest
                results = detector.backtest_signals(
                    epic_list=[test_epic],
                    lookback_days=days,
                    use_bid_adjustment=config.USE_BID_ADJUSTMENT,
                    spread_pips=config.SPREAD_PIPS,
                    timeframe='5m'
                )
                
                # Analyze performance
                from forex_scanner.backtests.performance_analyzer import PerformanceAnalyzer
                performance_analyzer = PerformanceAnalyzer()
                performance = performance_analyzer.analyze_performance(results)
                
                benchmark_results[combo_name] = performance
                
                signal_count = performance.get('total_signals', 0)
                avg_confidence = performance.get('average_confidence', 0)
                win_rate = performance.get('win_rate', 0)
                
                self.logger.info(f"   Results: {signal_count} signals, {avg_confidence:.1%} confidence, {win_rate:.1%} win rate")
            
            # Display benchmark comparison
            self._display_benchmark_results(benchmark_results)
            
            # Restore original settings
            config.ENABLE_BB_FILTER = original_bb_filter
            config.REQUIRE_VOLUME_CONFIRMATION = original_volume_req
            config.REQUIRE_EMA_SEPARATION = original_ema_sep
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark failed: {e}")
            return False
    
    def _display_benchmark_results(self, benchmark_results: Dict):
        """Display benchmark comparison results"""
        self.logger.info(f"\n⚡ INDICATOR PERFORMANCE BENCHMARK:")
        self.logger.info("=" * 85)
        
        # Header
        header = f"{'Combination':<15} {'Signals':<8} {'Avg Conf':<9} {'Win Rate':<9} {'Avg Profit':<11} {'Quality':<8}"
        self.logger.info(header)
        self.logger.info("-" * 85)
        
        # Sort by quality score
        sorted_results = sorted(
            benchmark_results.items(),
            key=lambda x: (x[1].get('average_confidence', 0) * 0.3 + 
                          x[1].get('win_rate', 0) * 0.7),
            reverse=True
        )
        
        for combo_name, performance in sorted_results:
            signal_count = performance.get('total_signals', 0)
            avg_confidence = performance.get('average_confidence', 0)
            win_rate = performance.get('win_rate', 0)
            avg_profit = performance.get('average_profit_pips', 0)
            
            # Quality score (weighted combination)
            quality_score = (avg_confidence * 0.3) + (win_rate * 0.7)
            
            row = (f"{combo_name:<15} "
                   f"{signal_count:<8} "
                   f"{avg_confidence:<9.1%} "
                   f"{win_rate:<9.1%} "
                   f"{avg_profit:<11.1f} "
                   f"{quality_score:<8.1%}")
            
            self.logger.info(row)
        
        self.logger.info("=" * 85)
        
        if sorted_results:
            best_combo = sorted_results[0]
            self.logger.info(f"🏆 Best performing combination: {best_combo[0]}")
            self.logger.info(f"💡 Consider using this configuration for optimal results")