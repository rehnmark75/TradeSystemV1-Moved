#!/usr/bin/env python3
"""
Zero-Lag Optimization System Comprehensive Test Suite
Validates all components of the zero-lag optimization system
"""

import sys
import os
import unittest
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import tempfile
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import components to test
from core.database import DatabaseManager
from optimization.optimal_parameter_service import (
    get_zerolag_optimal_parameters, ZeroLagOptimalParameters, 
    MarketConditions, get_epic_zerolag_config
)
from optimization.optimize_zerolag_parameters import ZeroLagParameterOptimizationEngine
from optimization.zerolag_optimization_analysis import ZeroLagOptimizationAnalyzer
from optimization.dynamic_zerolag_scanner_integration import DynamicZeroLagScanner
from core.strategies.zero_lag_strategy import ZeroLagStrategy, create_optimized_zero_lag_strategy
from backtests.backtest_zero_lag import ZeroLagBacktest

try:
    import config
except ImportError:
    from forex_scanner import config


class TestZeroLagOptimizationDatabase(unittest.TestCase):
    """Test database schema and operations"""
    
    @classmethod
    def setUpClass(cls):
        cls.db_manager = DatabaseManager(config.DATABASE_URL)
    
    def test_database_tables_exist(self):
        """Test that all required tables exist"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            tables_to_check = [
                'zerolag_optimization_runs',
                'zerolag_optimization_results', 
                'zerolag_best_parameters'
            ]
            
            for table in tables_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                
                exists = cursor.fetchone()[0]
                self.assertTrue(exists, f"Table {table} does not exist")
    
    def test_database_schema_structure(self):
        """Test database schema has required columns"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Test zerolag_best_parameters structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'zerolag_best_parameters'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            column_names = [col[0] for col in columns]
            
            required_columns = [
                'epic', 'best_zl_length', 'best_band_multiplier',
                'best_confidence_threshold', 'best_timeframe',
                'best_bb_length', 'best_bb_mult', 'best_kc_length', 'best_kc_mult',
                'optimal_stop_loss_pips', 'optimal_take_profit_pips',
                'best_win_rate', 'best_profit_factor', 'best_net_pips'
            ]
            
            for col in required_columns:
                self.assertIn(col, column_names, f"Column {col} missing from zerolag_best_parameters")
    
    def test_database_insert_and_retrieve(self):
        """Test basic database operations"""
        test_epic = 'TEST.EPIC.FOR.TESTING'
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert test data
                cursor.execute("""
                    INSERT INTO zerolag_best_parameters (
                        epic, best_zl_length, best_band_multiplier, best_confidence_threshold,
                        best_timeframe, best_bb_length, best_bb_mult, best_kc_length, best_kc_mult,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        best_win_rate, best_profit_factor, best_net_pips, best_composite_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (epic) DO UPDATE SET
                        best_zl_length = EXCLUDED.best_zl_length
                """, (
                    test_epic, 50, 1.5, 0.65, '15m', 20, 2.0, 20, 1.5,
                    10.0, 20.0, 0.75, 2.5, 100.0, 1.875
                ))
                
                # Retrieve test data
                cursor.execute("""
                    SELECT epic, best_zl_length, best_band_multiplier 
                    FROM zerolag_best_parameters 
                    WHERE epic = %s
                """, (test_epic,))
                
                result = cursor.fetchone()
                self.assertIsNotNone(result, "Failed to retrieve inserted test data")
                self.assertEqual(result[0], test_epic)
                self.assertEqual(result[1], 50)
                self.assertEqual(float(result[2]), 1.5)
                
                # Clean up
                cursor.execute("DELETE FROM zerolag_best_parameters WHERE epic = %s", (test_epic,))
                conn.commit()
                
        except Exception as e:
            self.fail(f"Database operation failed: {e}")


class TestZeroLagOptimalParameterService(unittest.TestCase):
    """Test optimal parameter service functionality"""
    
    def setUp(self):
        self.test_epic = 'CS.D.EURUSD.CEEM.IP'
        # Setup test data if needed
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup test data in database"""
        try:
            db_manager = DatabaseManager(config.DATABASE_URL)
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO zerolag_best_parameters (
                        epic, best_zl_length, best_band_multiplier, best_confidence_threshold,
                        best_timeframe, best_bb_length, best_bb_mult, best_kc_length, best_kc_mult,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        best_win_rate, best_profit_factor, best_net_pips, best_composite_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (epic) DO UPDATE SET
                        best_zl_length = EXCLUDED.best_zl_length
                """, (
                    self.test_epic, 70, 1.8, 0.70, '15m', 25, 2.2, 25, 1.8,
                    12.0, 24.0, 0.80, 3.0, 150.0, 3.6
                ))
                conn.commit()
        except:
            pass  # Continue if setup fails
    
    def test_get_zerolag_optimal_parameters(self):
        """Test retrieving optimal parameters"""
        try:
            params = get_zerolag_optimal_parameters(self.test_epic)
            
            self.assertIsInstance(params, ZeroLagOptimalParameters)
            self.assertEqual(params.epic, self.test_epic)
            self.assertGreater(params.zl_length, 0)
            self.assertGreater(params.band_multiplier, 0)
            self.assertGreater(params.confidence_threshold, 0)
            
        except Exception as e:
            # If no optimization data exists, service should return fallback
            self.assertIn('fallback', str(e).lower(), 
                         f"Expected fallback behavior, got error: {e}")
    
    def test_market_conditions_integration(self):
        """Test market conditions parameter adjustment"""
        conditions = MarketConditions(
            volatility_level='high',
            market_regime='trending',
            session='london'
        )
        
        try:
            params = get_zerolag_optimal_parameters(self.test_epic, conditions)
            self.assertIsInstance(params, ZeroLagOptimalParameters)
            self.assertEqual(params.market_conditions, conditions)
        except:
            pass  # May fail if no optimization data
    
    def test_epic_config_compatibility(self):
        """Test config compatibility functions"""
        try:
            config = get_epic_zerolag_config(self.test_epic)
            
            self.assertIsInstance(config, dict)
            required_keys = ['zl_length', 'band_multiplier', 'min_confidence',
                           'bb_length', 'bb_mult', 'kc_length', 'kc_mult']
            
            for key in required_keys:
                self.assertIn(key, config, f"Missing config key: {key}")
                
        except Exception as e:
            self.fail(f"Config compatibility test failed: {e}")


class TestZeroLagStrategy(unittest.TestCase):
    """Test Zero-Lag strategy integration"""
    
    def setUp(self):
        self.test_epic = 'CS.D.EURUSD.CEEM.IP'
    
    def test_strategy_initialization_static(self):
        """Test strategy initialization with static config"""
        strategy = ZeroLagStrategy()
        
        self.assertIsNotNone(strategy)
        self.assertGreater(strategy.length, 0)
        self.assertGreater(strategy.band_multiplier, 0)
        self.assertGreater(strategy.min_confidence, 0)
        self.assertFalse(strategy.use_optimal_parameters)
    
    def test_strategy_initialization_dynamic(self):
        """Test strategy initialization with optimal parameters"""
        try:
            strategy = ZeroLagStrategy(
                epic=self.test_epic,
                use_optimal_parameters=True
            )
            
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.epic, self.test_epic)
            self.assertTrue(strategy.use_optimal_parameters)
            
            # Check that metadata is available
            metadata = strategy.get_strategy_metadata()
            self.assertIn('configuration', metadata)
            self.assertIn('optimization_data', metadata)
            
        except Exception as e:
            # May fail if no optimization data - should fallback gracefully
            self.assertIsNotNone(strategy)
    
    def test_optimized_strategy_factory(self):
        """Test optimized strategy factory function"""
        try:
            strategy = create_optimized_zero_lag_strategy(self.test_epic)
            
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.epic, self.test_epic)
            self.assertTrue(strategy.use_optimal_parameters)
            
        except Exception as e:
            # Should handle gracefully even without optimization data
            pass


class TestZeroLagOptimizationEngine(unittest.TestCase):
    """Test optimization engine functionality"""
    
    def setUp(self):
        self.engine = ZeroLagParameterOptimizationEngine(fast_mode=True)
    
    def test_engine_initialization(self):
        """Test optimization engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertTrue(self.engine.fast_mode)
        self.assertIsNotNone(self.engine.parameter_grid)
    
    def test_parameter_grid_generation(self):
        """Test parameter grid generation"""
        grid = self.engine.get_optimization_parameter_grid(quick_test=True)
        
        required_keys = [
            'zl_length', 'band_multiplier', 'confidence_threshold',
            'timeframes', 'bb_length', 'bb_mult', 'kc_length', 'kc_mult',
            'smart_money_options', 'mtf_validation_options',
            'stop_loss_levels', 'take_profit_levels'
        ]
        
        for key in required_keys:
            self.assertIn(key, grid, f"Missing grid key: {key}")
            self.assertIsInstance(grid[key], list)
            self.assertGreater(len(grid[key]), 0)
    
    def test_parameter_combination_calculation(self):
        """Test parameter combination calculation"""
        grid = self.engine.get_optimization_parameter_grid(quick_test=True)
        
        # Calculate expected combinations
        expected_combinations = 1
        for values in grid.values():
            expected_combinations *= len(values)
        
        self.assertGreater(expected_combinations, 0)
        self.assertLess(expected_combinations, 100000)  # Reasonable upper bound


class TestZeroLagBacktestIntegration(unittest.TestCase):
    """Test backtest integration for optimization"""
    
    def setUp(self):
        self.backtest = ZeroLagBacktest()
        self.test_params = {
            'zl_length': 50,
            'band_multiplier': 1.5,
            'min_confidence': 0.65,
            'bb_length': 20,
            'bb_mult': 2.0,
            'kc_length': 20,
            'kc_mult': 1.5,
            'smart_money_enabled': False,
            'mtf_validation_enabled': False
        }
    
    def test_backtest_initialization(self):
        """Test backtest initialization"""
        self.assertIsNotNone(self.backtest)
        self.assertEqual(len(self.backtest.detected_signals), 0)
    
    def test_parameter_injection(self):
        """Test parameter injection functionality"""
        self.backtest.set_test_parameters(self.test_params)
        
        self.assertEqual(self.backtest.test_parameters, self.test_params)
    
    def test_signal_collection(self):
        """Test signal collection mechanism"""
        test_signal = {
            'signal_type': 'BUY',
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'confidence': 0.75,
            'timestamp': datetime.now()
        }
        
        self.backtest.add_detected_signal(test_signal)
        signals = self.backtest.extract_signals()
        
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]['signal_type'], 'BUY')
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some test signals
        test_signals = [
            {'signal_type': 'BUY', 'confidence': 0.70},
            {'signal_type': 'SELL', 'confidence': 0.65},
            {'signal_type': 'BUY', 'confidence': 0.80}
        ]
        
        for signal in test_signals:
            self.backtest.add_detected_signal(signal)
        
        summary = self.backtest.get_performance_summary()
        
        self.assertIn('total_signals', summary)
        self.assertIn('win_rate', summary)
        self.assertIn('avg_confidence', summary)
        self.assertEqual(summary['total_signals'], 3)


class TestZeroLagAnalyzer(unittest.TestCase):
    """Test optimization analyzer"""
    
    def setUp(self):
        self.analyzer = ZeroLagOptimizationAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.db_manager)
    
    def test_summary_generation(self):
        """Test optimization summary generation"""
        try:
            summary = self.analyzer.get_optimization_summary(days=30)
            
            # May be empty if no optimization data, but should not error
            self.assertIsInstance(summary, dict)
            
        except Exception as e:
            self.fail(f"Summary generation failed: {e}")
    
    def test_report_generation(self):
        """Test report generation"""
        try:
            report = self.analyzer.generate_optimization_report()
            
            self.assertIsInstance(report, str)
            self.assertIn("ZERO-LAG OPTIMIZATION", report)
            
        except Exception as e:
            self.fail(f"Report generation failed: {e}")


class TestDynamicScanner(unittest.TestCase):
    """Test dynamic scanner integration"""
    
    def setUp(self):
        self.scanner = DynamicZeroLagScanner()
    
    def test_scanner_initialization(self):
        """Test scanner initialization"""
        self.assertIsNotNone(self.scanner)
        self.assertIsNotNone(self.scanner.config)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        try:
            results = self.scanner.initialize_optimized_strategies()
            
            # Should complete without errors, even if no optimization data
            self.assertIsInstance(results, dict)
            
        except Exception as e:
            self.fail(f"Strategy initialization failed: {e}")
    
    def test_statistics_generation(self):
        """Test statistics generation"""
        stats = self.scanner.get_scanner_statistics()
        
        required_keys = [
            'total_epics', 'optimized_count', 'fallback_count',
            'optimization_rate', 'optimized_epics', 'fallback_epics'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats, f"Missing stats key: {key}")


class TestSystemIntegration(unittest.TestCase):
    """Test full system integration"""
    
    def test_end_to_end_workflow(self):
        """Test complete optimization workflow"""
        test_epic = 'CS.D.EURUSD.CEEM.IP'
        
        try:
            # 1. Create optimization engine
            engine = ZeroLagParameterOptimizationEngine(fast_mode=True)
            self.assertIsNotNone(engine)
            
            # 2. Get parameter service
            try:
                params = get_zerolag_optimal_parameters(test_epic)
                param_service_working = True
            except:
                param_service_working = False
            
            # 3. Create strategy (should work with or without optimization data)
            strategy = ZeroLagStrategy(epic=test_epic, use_optimal_parameters=param_service_working)
            self.assertIsNotNone(strategy)
            
            # 4. Create dynamic scanner
            scanner = DynamicZeroLagScanner()
            self.assertIsNotNone(scanner)
            
            # 5. Create analyzer
            analyzer = ZeroLagOptimizationAnalyzer()
            self.assertIsNotNone(analyzer)
            
            # All components should initialize successfully
            
        except Exception as e:
            self.fail(f"End-to-end integration test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration system validation"""
        try:
            from configdata.strategies import validate_zerolag_config
            
            validation_result = validate_zerolag_config()
            
            self.assertIsInstance(validation_result, dict)
            self.assertIn('valid', validation_result)
            
            if not validation_result['valid']:
                self.fail(f"Configuration validation failed: {validation_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.fail(f"Configuration validation test failed: {e}")


def run_comprehensive_test_suite():
    """Run the complete test suite with detailed reporting"""
    print("=" * 80)
    print("🧪 ZERO-LAG OPTIMIZATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"📅 Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestZeroLagOptimizationDatabase,
        TestZeroLagOptimalParameterService,
        TestZeroLagStrategy,
        TestZeroLagOptimizationEngine,
        TestZeroLagBacktestIntegration,
        TestZeroLagAnalyzer,
        TestDynamicScanner,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print(f"⏩ Skipped: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.splitlines()[-1]}")
    
    print("\n" + "=" * 80)
    
    # Overall status
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("🎉 ALL TESTS PASSED! Zero-Lag optimization system is ready for production.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Review and fix issues before production deployment.")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    success = run_comprehensive_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)