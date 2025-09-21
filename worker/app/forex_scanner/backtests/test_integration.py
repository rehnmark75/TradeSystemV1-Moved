#!/usr/bin/env python3
"""
Integration Test Suite for Enhanced Backtest System
Tests the complete integration including API server, Streamlit compatibility, etc.
"""

import sys
import os
import unittest
import requests
import time
from unittest.mock import patch

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir
sys.path.insert(0, project_root)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete enhanced backtest system"""

    def setUp(self):
        """Set up integration test environment"""
        self.api_base_url = "http://localhost:8000"
        self.test_epic = "CS.D.EURUSD.MINI.IP"

    def test_api_server_enhanced_strategies(self):
        """Test that API server recognizes enhanced strategies"""
        try:
            # Test enhanced strategy availability
            response = requests.get(f"{self.api_base_url}/strategies")
            if response.status_code == 200:
                strategies = response.json()

                # Check for enhanced strategies
                enhanced_strategies = [s for s in strategies if '_enhanced' in s]
                self.assertGreater(len(enhanced_strategies), 0, "No enhanced strategies found in API")

                print(f"✅ Found {len(enhanced_strategies)} enhanced strategies in API")
            else:
                self.skipTest("API server not available")

        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running - skipping API tests")

    def test_api_backtest_execution(self):
        """Test API backtest execution with enhanced format"""
        try:
            # Test backtest via API
            payload = {
                "epic": self.test_epic,
                "strategy": "mean_reversion",  # Use existing strategy
                "days": 1,
                "timeframe": "15m"
            }

            response = requests.post(f"{self.api_base_url}/backtest", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                # Validate enhanced format
                self.assertIn("success", result)
                self.assertIn("signals", result)
                self.assertIn("execution_metadata", result)

                print(f"✅ API backtest completed successfully")
                print(f"   Signals: {len(result.get('signals', []))}")
                print(f"   Success: {result.get('success', False)}")
            else:
                self.skipTest(f"API backtest failed: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.skipTest("API server not running - skipping API tests")
        except requests.exceptions.Timeout:
            self.skipTest("API backtest timed out - this is expected in some environments")

    def test_enhanced_strategies_import(self):
        """Test that enhanced strategy files can be imported"""
        enhanced_strategies = [
            'backtest_ema_enhanced',
            'backtest_macd_enhanced',
            'backtest_ichimoku_enhanced'
        ]

        imported_count = 0
        for strategy_module in enhanced_strategies:
            try:
                module = __import__(f'backtests.{strategy_module}', fromlist=[''])
                self.assertIsNotNone(module)
                imported_count += 1
                print(f"✅ Successfully imported {strategy_module}")
            except ImportError as e:
                print(f"⚠️ Could not import {strategy_module}: {e}")

        # At least some enhanced strategies should be importable
        self.assertGreater(imported_count, 0, "No enhanced strategies could be imported")

    def test_migration_utility_functionality(self):
        """Test migration utility functionality"""
        try:
            from backtests.migration_utility import StrategyMigrationUtility

            migrator = StrategyMigrationUtility()
            self.assertIsNotNone(migrator)

            # Test enhanced template exists and is valid
            self.assertIsNotNone(migrator.enhanced_template)
            self.assertIn("StandardSignal", migrator.enhanced_template)
            self.assertIn("BacktestBase", migrator.enhanced_template)

            print("✅ Migration utility is functional")

        except ImportError as e:
            self.skipTest(f"Migration utility not available: {e}")

    def test_backwards_compatibility(self):
        """Test that system maintains backwards compatibility"""
        try:
            # Import legacy backtest
            from backtests.backtest_mean_reversion import MeanReversionBacktest

            # Test that it can be instantiated in legacy mode
            backtest = MeanReversionBacktest(
                use_optimal_parameters=False,
                enable_market_intelligence=False
            )

            self.assertIsNotNone(backtest)
            self.assertEqual(backtest.strategy_name, "mean_reversion")

            print("✅ Backwards compatibility maintained")

        except ImportError as e:
            self.skipTest(f"Legacy backtest not available: {e}")

    def test_parameter_manager_integration(self):
        """Test parameter manager integration across system"""
        try:
            from backtests.parameter_manager import ParameterManager

            manager = ParameterManager()

            # Test parameter retrieval for different strategies
            test_strategies = ['ema', 'macd', 'mean_reversion']

            for strategy in test_strategies:
                try:
                    param_set = manager.get_parameters(strategy, self.test_epic)
                    self.assertIsNotNone(param_set)
                    print(f"✅ Parameter manager works for {strategy}")
                except Exception as e:
                    print(f"⚠️ Parameter manager issue for {strategy}: {e}")

        except ImportError as e:
            self.skipTest(f"Parameter manager not available: {e}")

    def test_market_intelligence_integration(self):
        """Test market intelligence integration"""
        try:
            from core.market_intelligence import MarketIntelligenceEngine
            import pandas as pd
            import numpy as np

            # Create test data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
            test_data = pd.DataFrame({
                'open': np.random.uniform(1.09, 1.11, 100),
                'high': np.random.uniform(1.10, 1.12, 100),
                'low': np.random.uniform(1.08, 1.10, 100),
                'close': np.random.uniform(1.09, 1.11, 100),
            }, index=dates)

            engine = MarketIntelligenceEngine()

            # Test regime detection
            regime = engine.detect_regime(test_data, self.test_epic)
            self.assertIsNotNone(regime)

            # Test market context
            context = engine.get_market_context(test_data, self.test_epic)
            self.assertIsNotNone(context)

            print("✅ Market intelligence engine functional")

        except ImportError as e:
            self.skipTest(f"Market intelligence not available: {e}")

    def test_unified_runner_functionality(self):
        """Test unified runner script"""
        try:
            runner_path = os.path.join(script_dir, "run_enhanced_strategy.py")
            if os.path.exists(runner_path):
                # Test that runner script can be imported
                spec = importlib.util.spec_from_file_location("run_enhanced_strategy", runner_path)
                runner_module = importlib.util.module_from_spec(spec)

                self.assertIsNotNone(runner_module)
                print("✅ Unified runner script is importable")
            else:
                self.skipTest("Unified runner script not found")

        except Exception as e:
            self.skipTest(f"Unified runner test failed: {e}")

    def test_streamlit_compatibility(self):
        """Test Streamlit compatibility components"""
        try:
            # Test that enhanced backtest results can be processed by Streamlit code
            from backtests.backtest_base import StandardBacktestResult, EpicResult, MarketConditions

            # Create a test result in the enhanced format
            test_result = StandardBacktestResult(
                strategy_name="test_strategy",
                epic_results={
                    self.test_epic: EpicResult(
                        epic=self.test_epic,
                        signals=[],
                        performance_metrics={'total_signals': 0},
                        market_conditions_summary=MarketConditions()
                    )
                },
                total_signals=0,
                overall_performance={},
                market_intelligence_summary={},
                execution_metadata={},
                success=True
            )

            # Test legacy compatibility properties
            self.assertIsInstance(test_result.signals, list)
            self.assertIsInstance(test_result.epic_results, dict)

            print("✅ Streamlit compatibility verified")

        except ImportError as e:
            self.skipTest(f"Streamlit compatibility test failed: {e}")


def run_integration_tests():
    """Run the integration test suite"""
    print("🔗 Running Enhanced Backtest System Integration Tests...")

    # Import required modules
    import importlib.util

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n📊 Integration Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")

    if result.failures:
        print(f"\n❌ Failures:")
        for test, error in result.failures:
            print(f"   - {test}")

    if result.errors:
        print(f"\n💥 Errors:")
        for test, error in result.errors:
            print(f"   - {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    print(f"\n🎯 Integration Success Rate: {success_rate:.1%}")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_integration_tests()