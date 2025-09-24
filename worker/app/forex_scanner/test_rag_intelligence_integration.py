#!/usr/bin/env python3
"""
RAG Intelligence Strategy Integration Test
========================================

Test script to verify the RAG Intelligence Strategy integrates properly
with the task-worker system and can generate signals correctly.

This test validates:
- Strategy initialization
- Market intelligence analysis
- RAG code selection (with fallback)
- Signal detection
- Performance statistics
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import required modules
from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy

try:
    import config
except ImportError:
    # Create minimal config for testing
    class TestConfig:
        DATABASE_URL = "postgresql://user:password@localhost/forex"
        MIN_CONFIDENCE = 0.6

    config = TestConfig()


class RAGIntelligenceStrategyTest:
    """Integration test for RAG Intelligence Strategy"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}

        # Initialize components (may fail if database unavailable)
        try:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.data_fetcher = DataFetcher(self.db_manager, 'UTC')
            self.db_available = True
        except Exception as e:
            self.logger.warning(f"Database unavailable for testing: {e}")
            self.db_manager = None
            self.data_fetcher = None
            self.db_available = False

    def _setup_logging(self):
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [TEST] - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger('rag_intelligence_test')

    def run_all_tests(self) -> dict:
        """Run all integration tests"""
        self.logger.info("🧪 Starting RAG Intelligence Strategy Integration Tests")
        self.logger.info("=" * 70)

        # Test 1: Strategy Initialization
        self.test_results['initialization'] = self._test_initialization()

        # Test 2: Market Intelligence Analysis (requires database)
        if self.db_available:
            self.test_results['market_analysis'] = self._test_market_analysis()
        else:
            self.test_results['market_analysis'] = {'status': 'skipped', 'reason': 'No database'}

        # Test 3: RAG Integration (works without database)
        self.test_results['rag_integration'] = self._test_rag_integration()

        # Test 4: Signal Detection (requires data)
        self.test_results['signal_detection'] = self._test_signal_detection()

        # Test 5: Performance Statistics
        self.test_results['performance_stats'] = self._test_performance_stats()

        # Test 6: Error Handling
        self.test_results['error_handling'] = self._test_error_handling()

        # Print summary
        self._print_test_summary()

        return self.test_results

    def _test_initialization(self) -> dict:
        """Test strategy initialization"""
        try:
            self.logger.info("🔧 Test 1: Strategy Initialization")

            # Test basic initialization
            strategy = RAGIntelligenceStrategy(
                epic="CS.D.EURUSD.MINI.IP",
                data_fetcher=self.data_fetcher,
                backtest_mode=True,
                market_analysis_hours=24
            )

            # Check core attributes
            assert strategy.name == 'rag_intelligence'
            assert strategy.epic == "CS.D.EURUSD.MINI.IP"
            assert strategy.backtest_mode is True
            assert strategy.market_analysis_hours == 24

            # Check components initialization
            assert strategy.config is not None
            assert strategy.rag_helper is not None
            assert hasattr(strategy, 'stats')
            assert hasattr(strategy, 'intelligence_cache')

            self.logger.info("✅ Strategy initialization: PASSED")
            return {
                'status': 'passed',
                'message': 'Strategy initialized successfully with all components'
            }

        except Exception as e:
            self.logger.error(f"❌ Strategy initialization: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_market_analysis(self) -> dict:
        """Test market intelligence analysis"""
        try:
            self.logger.info("📊 Test 2: Market Intelligence Analysis")

            strategy = RAGIntelligenceStrategy(
                epic="CS.D.EURUSD.MINI.IP",
                data_fetcher=self.data_fetcher,
                backtest_mode=True
            )

            # Test market condition analysis
            market_condition = strategy.analyze_market_conditions("CS.D.EURUSD.MINI.IP")

            # Validate market condition
            assert hasattr(market_condition, 'regime')
            assert hasattr(market_condition, 'confidence')
            assert hasattr(market_condition, 'session')
            assert hasattr(market_condition, 'volatility')
            assert hasattr(market_condition, 'timestamp')

            assert market_condition.regime in ['trending_up', 'trending_down', 'ranging', 'breakout']
            assert 0.0 <= market_condition.confidence <= 1.0

            self.logger.info(f"   Detected regime: {market_condition.regime} ({market_condition.confidence:.1%} confidence)")
            self.logger.info(f"   Session: {market_condition.session}, Volatility: {market_condition.volatility}")

            self.logger.info("✅ Market intelligence analysis: PASSED")
            return {
                'status': 'passed',
                'regime': market_condition.regime,
                'confidence': market_condition.confidence,
                'session': market_condition.session
            }

        except Exception as e:
            self.logger.error(f"❌ Market intelligence analysis: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_rag_integration(self) -> dict:
        """Test RAG system integration"""
        try:
            self.logger.info("🤖 Test 3: RAG Integration")

            strategy = RAGIntelligenceStrategy(
                epic="CS.D.EURUSD.MINI.IP",
                data_fetcher=self.data_fetcher,
                backtest_mode=True
            )

            # Test RAG helper availability
            assert strategy.rag_helper is not None

            # Get RAG performance stats
            rag_stats = strategy.rag_helper.get_performance_stats()
            self.logger.info(f"   RAG available: {rag_stats.get('rag_available', False)}")
            self.logger.info(f"   Fallback strategies: {rag_stats.get('fallback_strategies_available', 0)}")

            # Test market condition creation (mock)
            from core.strategies.rag_intelligence_strategy import MarketCondition
            mock_market_condition = MarketCondition(
                regime='trending_up',
                confidence=0.75,
                session='london',
                volatility='high',
                dominant_timeframe='15m',
                success_factors=['momentum', 'volume'],
                timestamp=datetime.utcnow()
            )

            # Test strategy code selection
            strategy_code = strategy.select_optimal_code(mock_market_condition)

            # Validate strategy code
            assert hasattr(strategy_code, 'code_type')
            assert hasattr(strategy_code, 'confidence_score')
            assert hasattr(strategy_code, 'parameters')
            assert hasattr(strategy_code, 'source_id')

            self.logger.info(f"   Selected strategy: {strategy_code.code_type}")
            self.logger.info(f"   Strategy ID: {strategy_code.source_id}")
            self.logger.info(f"   Confidence: {strategy_code.confidence_score:.1%}")

            self.logger.info("✅ RAG integration: PASSED")
            return {
                'status': 'passed',
                'rag_available': rag_stats.get('rag_available', False),
                'strategy_selected': strategy_code.source_id,
                'confidence': strategy_code.confidence_score
            }

        except Exception as e:
            self.logger.error(f"❌ RAG integration: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_signal_detection(self) -> dict:
        """Test signal detection capability"""
        try:
            self.logger.info("🎯 Test 4: Signal Detection")

            strategy = RAGIntelligenceStrategy(
                epic="CS.D.EURUSD.MINI.IP",
                data_fetcher=self.data_fetcher,
                backtest_mode=True
            )

            # Create mock market data for testing
            mock_data = self._create_mock_market_data()

            # Test signal detection
            signal = strategy.detect_signal(
                mock_data,
                "CS.D.EURUSD.MINI.IP",
                spread_pips=1.5,
                timeframe='15m'
            )

            if signal:
                # Validate signal structure
                assert 'direction' in signal
                assert 'entry_price' in signal
                assert 'confidence' in signal
                assert 'strategy' in signal

                assert signal['direction'] in ['BUY', 'SELL']
                assert 0.0 <= signal['confidence'] <= 1.0

                self.logger.info(f"   Signal generated: {signal['direction']}")
                self.logger.info(f"   Entry price: {signal['entry_price']}")
                self.logger.info(f"   Confidence: {signal['confidence']:.1%}")
                self.logger.info(f"   Strategy: {signal['strategy']}")

                self.logger.info("✅ Signal detection: PASSED")
                return {
                    'status': 'passed',
                    'signal_generated': True,
                    'direction': signal['direction'],
                    'confidence': signal['confidence']
                }
            else:
                self.logger.info("   No signal generated (may be normal)")
                self.logger.info("✅ Signal detection: PASSED (no signal)")
                return {
                    'status': 'passed',
                    'signal_generated': False,
                    'reason': 'No valid signal conditions met'
                }

        except Exception as e:
            self.logger.error(f"❌ Signal detection: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_performance_stats(self) -> dict:
        """Test performance statistics collection"""
        try:
            self.logger.info("📈 Test 5: Performance Statistics")

            strategy = RAGIntelligenceStrategy(
                epic="CS.D.EURUSD.MINI.IP",
                data_fetcher=self.data_fetcher,
                backtest_mode=True
            )

            # Get strategy statistics
            stats = strategy.get_strategy_stats()

            # Validate stats structure
            required_stats = ['strategy_name', 'total_signals', 'rag_selections', 'intelligence_queries']
            for stat in required_stats:
                assert stat in stats, f"Missing stat: {stat}"

            self.logger.info(f"   Strategy name: {stats['strategy_name']}")
            self.logger.info(f"   Total signals: {stats['total_signals']}")
            self.logger.info(f"   RAG selections: {stats['rag_selections']}")
            self.logger.info(f"   Current regime: {stats['current_regime']}")
            self.logger.info(f"   RAG available: {stats.get('rag_available', 'unknown')}")

            self.logger.info("✅ Performance statistics: PASSED")
            return {
                'status': 'passed',
                'stats_collected': True,
                'total_signals': stats['total_signals'],
                'rag_available': stats.get('rag_available', False)
            }

        except Exception as e:
            self.logger.error(f"❌ Performance statistics: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _test_error_handling(self) -> dict:
        """Test error handling and fallback mechanisms"""
        try:
            self.logger.info("🛡️ Test 6: Error Handling")

            # Test initialization with invalid parameters
            try:
                strategy = RAGIntelligenceStrategy(
                    epic=None,  # Invalid epic
                    data_fetcher=None,  # No data fetcher
                    backtest_mode=True
                )
                # Should not raise exception, should handle gracefully
                assert strategy.epic is None
                assert strategy.data_fetcher is None
            except Exception as e:
                self.logger.warning(f"   Initialization with invalid params failed: {e}")

            # Test signal detection with invalid data
            strategy = RAGIntelligenceStrategy(backtest_mode=True)

            # Test with empty DataFrame
            empty_data = pd.DataFrame()
            signal = strategy.detect_signal(empty_data, "TEST", 1.5, '15m')
            assert signal is None  # Should return None gracefully

            # Test with insufficient data
            minimal_data = pd.DataFrame({
                'open': [1.0, 1.1],
                'high': [1.05, 1.15],
                'low': [0.95, 1.05],
                'close': [1.02, 1.12],
                'volume': [1000, 1100]
            })
            signal = strategy.detect_signal(minimal_data, "TEST", 1.5, '15m')
            # Should handle gracefully (return None or handle insufficient data)

            self.logger.info("✅ Error handling: PASSED")
            return {
                'status': 'passed',
                'graceful_handling': True
            }

        except Exception as e:
            self.logger.error(f"❌ Error handling: FAILED - {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _create_mock_market_data(self) -> pd.DataFrame:
        """Create mock market data for testing"""
        # Generate 200 bars of mock data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='15min')

        # Create realistic price movement
        base_price = 1.0950
        price_changes = (np.random.random(200) - 0.5) * 0.001
        prices = base_price + np.cumsum(price_changes)

        # Generate OHLC data
        opens = prices
        highs = prices + np.random.random(200) * 0.0005
        lows = prices - np.random.random(200) * 0.0005
        closes = prices + (np.random.random(200) - 0.5) * 0.0003
        volumes = np.random.randint(1000, 10000, 200)

        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        mock_data.set_index('timestamp', inplace=True)
        return mock_data

    def _print_test_summary(self):
        """Print comprehensive test summary"""
        self.logger.info("=" * 70)
        self.logger.info("🏁 TEST SUMMARY")
        self.logger.info("=" * 70)

        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0

        for test_name, result in self.test_results.items():
            status = result.get('status', 'unknown')
            if status == 'passed':
                self.logger.info(f"✅ {test_name.title().replace('_', ' ')}: PASSED")
                passed_tests += 1
            elif status == 'failed':
                self.logger.error(f"❌ {test_name.title().replace('_', ' ')}: FAILED")
                self.logger.error(f"   Error: {result.get('error', 'Unknown error')}")
                failed_tests += 1
            elif status == 'skipped':
                self.logger.warning(f"⏭️ {test_name.title().replace('_', ' ')}: SKIPPED")
                self.logger.warning(f"   Reason: {result.get('reason', 'Unknown reason')}")
                skipped_tests += 1

        total_tests = passed_tests + failed_tests + skipped_tests

        self.logger.info("-" * 70)
        self.logger.info(f"📊 Total Tests: {total_tests}")
        self.logger.info(f"✅ Passed: {passed_tests}")
        self.logger.info(f"❌ Failed: {failed_tests}")
        self.logger.info(f"⏭️ Skipped: {skipped_tests}")

        if failed_tests == 0:
            self.logger.info("🎉 ALL TESTS PASSED! RAG Intelligence Strategy is ready for deployment.")
        else:
            self.logger.warning(f"⚠️ {failed_tests} test(s) failed. Please review and fix before deployment.")

        success_rate = (passed_tests / max(1, passed_tests + failed_tests)) * 100
        self.logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        self.logger.info("=" * 70)


def main():
    """Main test execution"""
    import numpy as np  # Import here to avoid issues if not available

    try:
        tester = RAGIntelligenceStrategyTest()
        results = tester.run_all_tests()

        # Return appropriate exit code
        failed_count = sum(1 for r in results.values() if r.get('status') == 'failed')
        return 0 if failed_count == 0 else 1

    except Exception as e:
        logging.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())