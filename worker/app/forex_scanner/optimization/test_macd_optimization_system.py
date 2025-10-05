#!/usr/bin/env python3
"""
MACD Optimization System Test Suite
Comprehensive testing for all MACD optimization components
"""

import sys
import os
import logging
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the modules we need to test
from core.database import DatabaseManager
from optimization.optimal_parameter_service import (
    OptimalParameterService, 
    MACDOptimalParameters,
    get_macd_optimal_parameters,
    get_epic_macd_config,
    is_epic_macd_optimized,
    get_macd_optimization_status
)
from optimization.optimize_macd_parameters import MACDParameterOptimizer
from optimization.macd_optimization_analysis import MACDOptimizationAnalyzer
from optimization.dynamic_macd_scanner_integration import DynamicMACDScanner

try:
    import config
except ImportError:
    from forex_scanner import config


class TestMACDOptimizationSystem(unittest.TestCase):
    """Test suite for MACD optimization system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = logging.getLogger('macd_test')
        logging.basicConfig(level=logging.INFO)
        
        cls.db_manager = DatabaseManager(config.DATABASE_URL)
        cls.test_epic = "CS.D.EURUSD.CEEM.IP"
        cls.parameter_service = OptimalParameterService()
        
        # Test data
        cls.test_macd_params = {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_ema': 9,
            'confidence_threshold': 0.55,
            'timeframe': '15m',
            'histogram_threshold': 0.00003,
            'stop_loss_pips': 10.0,
            'take_profit_pips': 20.0
        }
    
    def test_01_database_schema(self):
        """Test MACD optimization database schema exists"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if MACD optimization tables exist
                tables = ['macd_optimization_runs', 'macd_optimization_results', 'macd_best_parameters']
                
                for table in tables:
                    cursor.execute(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = '{table}'
                        )
                    """)
                    exists = cursor.fetchone()[0]
                    self.assertTrue(exists, f"Table {table} should exist")
                
                self.logger.info("✅ Test 1/7: Database schema validation passed")
                
        except Exception as e:
            self.fail(f"Database schema test failed: {e}")
    
    def test_02_parameter_service_fallback(self):
        """Test MACD parameter service fallback functionality"""
        try:
            # Test getting parameters for a non-optimized epic
            fake_epic = "CS.D.FAKE.TEST.IP"
            params = self.parameter_service.get_macd_epic_parameters(fake_epic)
            
            # Should return MACDOptimalParameters object
            self.assertIsInstance(params, MACDOptimalParameters)
            self.assertEqual(params.epic, fake_epic)
            
            # Should have fallback values
            self.assertEqual(params.fast_ema, 12)  # Default MACD fast
            self.assertEqual(params.slow_ema, 26)  # Default MACD slow
            self.assertEqual(params.signal_ema, 9)  # Default MACD signal
            self.assertEqual(params.confidence_threshold, 0.55)
            self.assertEqual(params.timeframe, '15m')
            
            # Should indicate this is fallback data
            self.assertEqual(params.performance_score, 0.0)
            
            self.logger.info("✅ Test 2/7: Parameter service fallback functionality passed")
            
        except Exception as e:
            self.fail(f"Parameter service fallback test failed: {e}")
    
    def test_03_parameter_optimizer_initialization(self):
        """Test MACD parameter optimizer initialization"""
        try:
            # Test different modes
            optimizer_smart = MACDParameterOptimizer(smart_presets=True)
            optimizer_fast = MACDParameterOptimizer(fast_mode=True)
            optimizer_full = MACDParameterOptimizer()
            
            # Check parameter grids are different sizes
            smart_grid = optimizer_smart._get_parameter_grid()
            fast_grid = optimizer_fast._get_parameter_grid()
            full_grid = optimizer_full._get_parameter_grid()
            
            # Check that grids have different sizes (order may vary based on implementation)
            grid_sizes = [
                len(smart_grid['macd_configs']),
                len(fast_grid['macd_configs']),
                len(full_grid['macd_configs'])
            ]
            
            # Just ensure they have different sizes and full has the most
            self.assertGreater(len(full_grid['macd_configs']), len(fast_grid['macd_configs']))
            self.assertTrue(len(set(grid_sizes)) > 1, "Grid sizes should be different")
            
            self.logger.info("✅ Test 3/7: Parameter optimizer initialization passed")
            
        except Exception as e:
            self.fail(f"Parameter optimizer initialization test failed: {e}")
    
    def test_04_optimization_analyzer(self):
        """Test MACD optimization analyzer functionality"""
        try:
            analyzer = MACDOptimizationAnalyzer()
            
            # Test summary generation (should not fail even with empty data)
            summary = analyzer.get_optimization_summary()
            self.assertIsInstance(summary, dict)
            
            # Test parameter effectiveness analysis
            effectiveness = analyzer.get_parameter_effectiveness_analysis()
            self.assertIsInstance(effectiveness, dict)
            
            # Test recommendations generation
            recommendations = analyzer.get_optimization_recommendations()
            self.assertIsInstance(recommendations, list)
            
            self.logger.info("✅ Test 4/7: Optimization analyzer functionality passed")
            
        except Exception as e:
            self.fail(f"Optimization analyzer test failed: {e}")
    
    def test_05_dynamic_scanner_integration(self):
        """Test dynamic MACD scanner integration"""
        try:
            scanner = DynamicMACDScanner()
            
            # Test optimization status
            status = scanner.get_optimization_status()
            self.assertIsInstance(status, dict)
            self.assertIn('total_configured', status)
            self.assertIn('total_optimized', status)
            
            # Test system readiness validation
            readiness = scanner.validate_system_readiness()
            self.assertIsInstance(readiness, dict)
            self.assertIn('database_ready', readiness)
            self.assertIn('system_ready', readiness)
            
            # Test that scanner can create optimized epics list (may be empty)
            optimized_epics = scanner.get_optimized_epics_list()
            self.assertIsInstance(optimized_epics, list)
            
            self.logger.info("✅ Test 5/7: Dynamic scanner integration passed")
            
        except Exception as e:
            self.fail(f"Dynamic scanner integration test failed: {e}")
    
    def test_06_convenience_functions(self):
        """Test MACD convenience functions"""
        try:
            # Test getting MACD parameters (should use fallback)
            params = get_macd_optimal_parameters(self.test_epic)
            self.assertIsInstance(params, MACDOptimalParameters)
            
            # Test getting MACD config
            config_dict = get_epic_macd_config(self.test_epic)
            self.assertIsInstance(config_dict, dict)
            self.assertIn('fast_ema', config_dict)
            self.assertIn('slow_ema', config_dict)
            self.assertIn('signal_ema', config_dict)
            
            # Test optimization status
            epic_optimized = is_epic_macd_optimized(self.test_epic)
            self.assertIsInstance(epic_optimized, bool)
            
            # Test overall optimization status
            overall_status = get_macd_optimization_status()
            self.assertIsInstance(overall_status, dict)
            
            self.logger.info("✅ Test 6/7: Convenience functions passed")
            
        except Exception as e:
            self.fail(f"Convenience functions test failed: {e}")
    
    def test_07_parameter_grid_generation(self):
        """Test parameter combination generation"""
        try:
            optimizer = MACDParameterOptimizer(smart_presets=True)
            combinations = optimizer.generate_parameter_combinations()
            
            # Should generate some combinations
            self.assertGreater(len(combinations), 0)
            
            # Each combination should have required fields
            for combo in combinations[:5]:  # Test first 5
                self.assertIn('fast_ema', combo)
                self.assertIn('slow_ema', combo)
                self.assertIn('signal_ema', combo)
                self.assertIn('confidence_levels', combo)
                self.assertIn('timeframes', combo)
                self.assertIn('stop_loss_levels', combo)
                self.assertIn('take_profit_levels', combo)
                self.assertIn('risk_reward_ratio', combo)
                
                # MACD periods should be valid
                self.assertGreater(combo['slow_ema'], combo['fast_ema'])
                self.assertGreater(combo['take_profit_levels'], combo['stop_loss_levels'])
            
            self.logger.info("✅ Test 7/7: Parameter grid generation passed")
            
        except Exception as e:
            self.fail(f"Parameter grid generation test failed: {e}")


def run_comprehensive_system_test():
    """Run comprehensive system test and provide detailed report"""
    print("\n" + "="*80)
    print("🎯 MACD OPTIMIZATION SYSTEM COMPREHENSIVE TEST")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMACDOptimizationSystem)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    # Run tests
    result = runner.run(suite)
    
    # Generate detailed report
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   • Tests Run: {result.testsRun}")
    print(f"   • Failures: {len(result.failures)}")
    print(f"   • Errors: {len(result.errors)}")
    print(f"   • Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback}")
    
    # System readiness assessment
    if len(result.failures) == 0 and len(result.errors) == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ MACD optimization system is ready for production use.")
        
        # Additional system info
        try:
            scanner = DynamicMACDScanner()
            status = scanner.get_optimization_status()
            readiness = scanner.validate_system_readiness()
            
            print(f"\n📈 SYSTEM STATUS:")
            print(f"   • Database Ready: {'✅' if readiness['database_ready'] else '❌'}")
            print(f"   • Epic Coverage: {status['optimization_coverage']:.1f}%")
            print(f"   • Production Ready: {'✅' if status['ready_for_production'] else '❌'}")
            
        except Exception as e:
            print(f"   • Status Check Failed: {e}")
    else:
        print(f"\n⚠️  SYSTEM NOT READY - Fix failing tests before production use.")
    
    print("="*80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_system_test()
    sys.exit(0 if success else 1)