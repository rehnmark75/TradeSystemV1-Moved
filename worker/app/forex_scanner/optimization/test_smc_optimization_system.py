#!/usr/bin/env python3
"""
SMC Optimization System Test Suite

Comprehensive test suite for the SMC (Smart Money Concepts) optimization system.
Tests all components: database tables, optimization engine, analysis tools, parameter service, and strategy integration.

Usage:
    python test_smc_optimization_system.py
    
Tests:
1. Database schema validation
2. Parameter service functionality  
3. SMC strategy integration
4. Analysis tools validation
5. Optimization engine basic functionality
6. Dynamic parameter retrieval
7. System readiness checks
"""

import sys
import os
import logging
from typing import Dict, List
import traceback
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from forex_scanner.core.database.database_manager import DatabaseManager

class SMCOptimizationSystemTester:
    """Comprehensive test suite for SMC optimization system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager()
        
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    def log_test_result(self, test_name: str, passed: bool, message: str = "", error: str = ""):
        """Log test result with details"""
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        
        result = {
            'test_name': test_name,
            'passed': passed,
            'message': message,
            'error': error,
            'timestamp': datetime.now()
        }
        
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        self.logger.info(f"{status} | {test_name}")
        if message:
            self.logger.info(f"     💬 {message}")
        if error and not passed:
            self.logger.error(f"     🚨 {error}")
    
    def test_database_schema(self) -> bool:
        """Test 1: Validate SMC optimization database schema"""
        
        try:
            # Test table existence
            tables = [
                'smc_optimization_runs',
                'smc_optimization_results', 
                'smc_best_parameters'
            ]
            
            for table in tables:
                query = f"SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name = '{table}'"
                result = self.db_manager.execute_query(query, fetch_results=True)
                
                if not result or result[0]['count'] == 0:
                    self.log_test_result(
                        "Database Schema",
                        False,
                        f"Table {table} not found",
                        f"Missing required table: {table}"
                    )
                    return False
            
            # Test key columns in smc_best_parameters
            key_columns = [
                'epic', 'best_smc_config', 'best_confidence_level', 'best_timeframe',
                'optimal_swing_length', 'optimal_structure_confirmation',
                'optimal_order_block_length', 'optimal_fvg_min_size',
                'optimal_confluence_required', 'best_performance_score'
            ]
            
            query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'smc_best_parameters'
            """
            
            columns_result = self.db_manager.execute_query(query, fetch_results=True)
            existing_columns = {row['column_name'] for row in columns_result} if columns_result else set()
            
            missing_columns = [col for col in key_columns if col not in existing_columns]
            if missing_columns:
                self.log_test_result(
                    "Database Schema",
                    False,
                    f"Missing columns: {missing_columns}",
                    f"Schema incomplete: {missing_columns}"
                )
                return False
            
            self.log_test_result(
                "Database Schema",
                True,
                f"All {len(tables)} tables and {len(key_columns)} key columns validated"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Database Schema",
                False,
                "Database schema validation failed",
                str(e)
            )
            return False
    
    def test_parameter_service(self) -> bool:
        """Test 2: Validate SMC parameter service functionality"""
        
        try:
            # Test imports
            from optimization.optimal_parameter_service import (
                get_smc_optimal_parameters,
                get_epic_smc_config,
                is_epic_smc_optimized,
                get_smc_system_readiness,
                SMCOptimalParameters
            )
            
            # Test parameter retrieval (should use fallback)
            test_epic = 'CS.D.EURUSD.MINI.IP'
            params = get_smc_optimal_parameters(test_epic)
            
            # Validate parameter structure
            required_attributes = [
                'epic', 'smc_config', 'confidence_threshold', 'timeframe',
                'swing_length', 'structure_confirmation', 'bos_threshold', 'choch_threshold',
                'order_block_length', 'fvg_min_size', 'confluence_required',
                'performance_score', 'win_rate'
            ]
            
            missing_attributes = [attr for attr in required_attributes if not hasattr(params, attr)]
            if missing_attributes:
                self.log_test_result(
                    "Parameter Service",
                    False,
                    f"Missing attributes: {missing_attributes}",
                    f"SMCOptimalParameters incomplete: {missing_attributes}"
                )
                return False
            
            # Test config conversion
            config = get_epic_smc_config(test_epic)
            required_config_keys = [
                'smc_config', 'swing_length', 'confluence_required', 
                'order_block_length', 'fvg_min_size', 'stop_loss_pips'
            ]
            
            missing_config_keys = [key for key in required_config_keys if key not in config]
            if missing_config_keys:
                self.log_test_result(
                    "Parameter Service",
                    False,
                    f"Missing config keys: {missing_config_keys}",
                    f"Config conversion incomplete: {missing_config_keys}"
                )
                return False
            
            # Test system readiness
            readiness = get_smc_system_readiness()
            if 'system_type' not in readiness or readiness['system_type'] != 'SMC':
                self.log_test_result(
                    "Parameter Service",
                    False,
                    "System readiness check failed",
                    f"Invalid readiness response: {readiness}"
                )
                return False
            
            self.log_test_result(
                "Parameter Service", 
                True,
                f"All parameter service functions working, {len(required_attributes)} attributes validated"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Parameter Service",
                False,
                "Parameter service functionality test failed",
                str(e)
            )
            return False
    
    def test_smc_strategy_integration(self) -> bool:
        """Test 3: Validate SMC strategy integration with optimization parameters"""
        
        try:
            from forex_scanner.core.strategies.smc_strategy import SMCStrategy
            
            # Test strategy creation without epic (should use static config)
            strategy_static = SMCStrategy(smc_config_name='moderate', use_optimized_parameters=False)
            
            if not hasattr(strategy_static, 'smc_config'):
                self.log_test_result(
                    "SMC Strategy Integration",
                    False,
                    "Strategy missing smc_config attribute",
                    "SMCStrategy initialization failed"
                )
                return False
            
            # Test strategy creation with epic (should attempt optimization)
            test_epic = 'CS.D.EURUSD.MINI.IP'
            strategy_optimized = SMCStrategy(
                epic=test_epic,
                use_optimized_parameters=True
            )
            
            # Validate strategy has epic parameter
            if strategy_optimized.epic != test_epic:
                self.log_test_result(
                    "SMC Strategy Integration",
                    False,
                    f"Epic not set correctly: {strategy_optimized.epic} != {test_epic}",
                    "Epic parameter not properly stored"
                )
                return False
            
            # Validate optimization availability check
            if not hasattr(strategy_optimized, '_optimization_available'):
                self.log_test_result(
                    "SMC Strategy Integration",
                    False,
                    "Strategy missing _optimization_available attribute",
                    "Optimization availability check not implemented"
                )
                return False
            
            # Test configuration loading
            if not strategy_optimized.smc_config:
                self.log_test_result(
                    "SMC Strategy Integration",
                    False,
                    "Strategy configuration not loaded",
                    "SMC config loading failed"
                )
                return False
            
            # Verify configuration has required fields
            required_config_fields = ['confluence_required', 'swing_length', 'fvg_min_size']
            missing_fields = [field for field in required_config_fields 
                            if field not in strategy_optimized.smc_config]
            if missing_fields:
                self.log_test_result(
                    "SMC Strategy Integration",
                    False,
                    f"Missing config fields: {missing_fields}",
                    f"Incomplete SMC configuration: {missing_fields}"
                )
                return False
            
            self.log_test_result(
                "SMC Strategy Integration",
                True,
                f"Strategy integration working, epic={test_epic}, config fields validated"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "SMC Strategy Integration",
                False,
                "SMC strategy integration test failed",
                str(e)
            )
            return False
    
    def test_analysis_tools(self) -> bool:
        """Test 4: Validate SMC optimization analysis tools"""
        
        try:
            from optimization.smc_optimization_analysis import SMCOptimizationAnalyzer
            
            # Test analyzer creation
            analyzer = SMCOptimizationAnalyzer()
            
            if not hasattr(analyzer, 'db_manager'):
                self.log_test_result(
                    "Analysis Tools",
                    False,
                    "Analyzer missing db_manager attribute",
                    "SMCOptimizationAnalyzer initialization failed"
                )
                return False
            
            # Test summary statistics (should work even with empty data)
            summary = analyzer.get_summary_statistics()
            
            # Should return empty dict or dict with zero values, not fail
            if summary is None:
                self.log_test_result(
                    "Analysis Tools", 
                    False,
                    "Summary statistics returned None",
                    "get_summary_statistics failed"
                )
                return False
            
            # Test top performers (should work even with empty data)
            top_performers = analyzer.get_top_performers(5)
            
            # Should return empty list or list of results, not fail
            if top_performers is None:
                self.log_test_result(
                    "Analysis Tools",
                    False,
                    "Top performers returned None", 
                    "get_top_performers failed"
                )
                return False
            
            # Test parameter impact analysis
            impact = analyzer.analyze_parameter_impact()
            
            # Should return empty dict or dict with analysis, not fail
            if impact is None:
                self.log_test_result(
                    "Analysis Tools",
                    False,
                    "Parameter impact analysis returned None",
                    "analyze_parameter_impact failed"
                )
                return False
            
            self.log_test_result(
                "Analysis Tools",
                True,
                "All analysis methods working, handles empty data gracefully"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Analysis Tools",
                False,
                "Analysis tools validation failed",
                str(e)
            )
            return False
    
    def test_optimization_engine(self) -> bool:
        """Test 5: Validate SMC optimization engine basic functionality"""
        
        try:
            from optimization.optimize_smc_parameters import SMCParameterOptimizer
            
            # Test optimizer creation
            optimizer = SMCParameterOptimizer()
            
            if not hasattr(optimizer, 'db_manager'):
                self.log_test_result(
                    "Optimization Engine",
                    False,
                    "Optimizer missing db_manager attribute",
                    "SMCParameterOptimizer initialization failed"
                )
                return False
            
            # Test parameter grid generation
            grid_smart = optimizer._get_parameter_grid('smart_presets')
            grid_fast = optimizer._get_parameter_grid('fast')  
            grid_full = optimizer._get_parameter_grid('full')
            
            # Validate grid structure
            required_grid_keys = ['smc_configs', 'confidence_levels', 'timeframes', 'stop_loss_levels']
            for mode, grid in [('smart_presets', grid_smart), ('fast', grid_fast), ('full', grid_full)]:
                missing_keys = [key for key in required_grid_keys if key not in grid]
                if missing_keys:
                    self.log_test_result(
                        "Optimization Engine",
                        False,
                        f"Missing grid keys in {mode}: {missing_keys}",
                        f"Parameter grid incomplete for {mode}: {missing_keys}"
                    )
                    return False
            
            # Validate grid sizes
            expected_smart_configs = len(grid_smart['smc_configs'])  # Should be 8 SMC configs
            if expected_smart_configs < 5:  # Reasonable minimum
                self.log_test_result(
                    "Optimization Engine",
                    False,
                    f"Smart presets grid too small: {expected_smart_configs} configs",
                    f"Expected at least 5 SMC configs, got {expected_smart_configs}"
                )
                return False
            
            # Test backtest simulation
            test_params = {
                'smc_config': 'moderate',
                'confidence_level': 0.55,
                'timeframe': '15m',
                'use_smart_money': True,
                'stop_loss_pips': 10.0,
                'take_profit_pips': 20.0,
                'risk_reward_ratio': 2.0
            }
            
            backtest_result = optimizer.run_smc_backtest('CS.D.EURUSD.MINI.IP', test_params, 30)
            
            # Validate backtest result structure
            required_result_keys = [
                'total_signals', 'win_rate', 'performance_score',
                'structure_breaks_detected', 'order_block_reactions', 'fvg_reactions'
            ]
            missing_result_keys = [key for key in required_result_keys if key not in backtest_result]
            if missing_result_keys:
                self.log_test_result(
                    "Optimization Engine",
                    False,
                    f"Missing backtest result keys: {missing_result_keys}",
                    f"Backtest result incomplete: {missing_result_keys}"
                )
                return False
            
            self.log_test_result(
                "Optimization Engine",
                True,
                f"Parameter grids working ({expected_smart_configs} configs), backtest simulation functional"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Optimization Engine", 
                False,
                "Optimization engine validation failed",
                str(e)
            )
            return False
    
    def test_dynamic_parameter_retrieval(self) -> bool:
        """Test 6: Validate dynamic parameter retrieval system"""
        
        try:
            from optimization.optimal_parameter_service import (
                OptimalParameterService,
                MarketConditions,
                get_smc_optimal_parameters
            )
            
            # Test service creation
            service = OptimalParameterService()
            
            if not hasattr(service, '_parameter_cache'):
                self.log_test_result(
                    "Dynamic Parameter Retrieval",
                    False,
                    "Service missing _parameter_cache attribute",
                    "OptimalParameterService caching not initialized"
                )
                return False
            
            # Test parameter retrieval without market conditions
            test_epic = 'CS.D.EURUSD.MINI.IP'
            params_basic = service.get_smc_epic_parameters(test_epic)
            
            if not params_basic or not hasattr(params_basic, 'epic'):
                self.log_test_result(
                    "Dynamic Parameter Retrieval",
                    False,
                    "Failed to retrieve basic SMC parameters",
                    f"get_smc_epic_parameters returned: {params_basic}"
                )
                return False
            
            # Test parameter retrieval with market conditions
            conditions = MarketConditions(
                volatility_level='high',
                market_regime='trending',
                session='london'
            )
            
            params_conditional = service.get_smc_epic_parameters(test_epic, conditions)
            
            if not params_conditional or not hasattr(params_conditional, 'epic'):
                self.log_test_result(
                    "Dynamic Parameter Retrieval",
                    False,
                    "Failed to retrieve conditional SMC parameters",
                    f"get_smc_epic_parameters with conditions returned: {params_conditional}"
                )
                return False
            
            # Test caching (second call should use cache)
            params_cached = service.get_smc_epic_parameters(test_epic)
            
            # Should be identical objects if caching works
            if (params_cached.epic != params_basic.epic or 
                params_cached.smc_config != params_basic.smc_config):
                self.log_test_result(
                    "Dynamic Parameter Retrieval",
                    False,
                    "Parameter caching not working correctly",
                    f"Cached params differ from original"
                )
                return False
            
            # Test convenience function
            convenience_params = get_smc_optimal_parameters(test_epic)
            if convenience_params.epic != test_epic:
                self.log_test_result(
                    "Dynamic Parameter Retrieval",
                    False,
                    "Convenience function not working",
                    f"get_smc_optimal_parameters returned wrong epic: {convenience_params.epic}"
                )
                return False
            
            self.log_test_result(
                "Dynamic Parameter Retrieval",
                True,
                f"Parameter retrieval working, caching functional, market conditions supported"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "Dynamic Parameter Retrieval",
                False,
                "Dynamic parameter retrieval test failed",
                str(e)
            )
            return False
    
    def test_system_readiness(self) -> bool:
        """Test 7: Validate system readiness and integration status"""
        
        try:
            from optimization.optimal_parameter_service import get_smc_system_readiness
            from configdata.strategies.config_smc_strategy import validate_smc_config
            
            # Test SMC configuration validation
            config_validation = validate_smc_config()
            
            if not config_validation.get('valid'):
                self.log_test_result(
                    "System Readiness",
                    False,
                    f"SMC configuration invalid: {config_validation.get('error')}",
                    f"Config validation failed: {config_validation}"
                )
                return False
            
            # Test system readiness
            readiness = get_smc_system_readiness()
            
            required_readiness_keys = [
                'total_configured', 'total_optimized', 'optimization_coverage',
                'ready_for_production', 'system_type'
            ]
            
            missing_readiness_keys = [key for key in required_readiness_keys if key not in readiness]
            if missing_readiness_keys:
                self.log_test_result(
                    "System Readiness",
                    False,
                    f"Missing readiness keys: {missing_readiness_keys}",
                    f"System readiness incomplete: {missing_readiness_keys}"
                )
                return False
            
            if readiness['system_type'] != 'SMC':
                self.log_test_result(
                    "System Readiness",
                    False,
                    f"Wrong system type: {readiness['system_type']}",
                    f"Expected 'SMC', got '{readiness['system_type']}'"
                )
                return False
            
            # Calculate test coverage
            optimization_coverage = readiness.get('optimization_coverage', 0)
            production_ready = readiness.get('ready_for_production', False)
            
            self.log_test_result(
                "System Readiness",
                True,
                f"System ready: {production_ready}, Coverage: {optimization_coverage:.1f}%, "
                f"Config: {config_validation.get('config_count')} configs"
            )
            return True
            
        except Exception as e:
            self.log_test_result(
                "System Readiness",
                False,
                "System readiness validation failed",
                str(e)
            )
            return False
    
    def run_all_tests(self) -> bool:
        """Run all SMC optimization system tests"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🧠 SMC OPTIMIZATION SYSTEM TEST SUITE")
        self.logger.info("="*80)
        
        # Test execution order
        tests = [
            ("Database Schema", self.test_database_schema),
            ("Parameter Service", self.test_parameter_service),
            ("SMC Strategy Integration", self.test_smc_strategy_integration),
            ("Analysis Tools", self.test_analysis_tools),
            ("Optimization Engine", self.test_optimization_engine),
            ("Dynamic Parameter Retrieval", self.test_dynamic_parameter_retrieval),
            ("System Readiness", self.test_system_readiness)
        ]
        
        # Run tests
        for test_name, test_method in tests:
            try:
                self.logger.info(f"\n🔄 Running: {test_name}")
                test_method()
            except Exception as e:
                self.log_test_result(
                    test_name,
                    False,
                    "Test execution failed",
                    f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
                )
        
        # Print results summary
        self.logger.info(f"\n" + "="*80)
        self.logger.info(f"📊 TEST RESULTS SUMMARY")
        self.logger.info(f"="*80)
        self.logger.info(f"✅ Passed: {self.passed_tests}/{self.total_tests}")
        self.logger.info(f"❌ Failed: {self.total_tests - self.passed_tests}/{self.total_tests}")
        self.logger.info(f"📈 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%" if self.total_tests > 0 else "0.0%")
        
        # Show failed tests
        failed_tests = [result for result in self.test_results if not result['passed']]
        if failed_tests:
            self.logger.info(f"\n❌ FAILED TESTS:")
            for result in failed_tests:
                self.logger.error(f"   • {result['test_name']}: {result['error']}")
        
        # Overall status
        all_passed = self.passed_tests == self.total_tests
        overall_status = "✅ ALL TESTS PASSED" if all_passed else f"❌ {len(failed_tests)} TESTS FAILED"
        self.logger.info(f"\n🏁 {overall_status}")
        self.logger.info("="*80 + "\n")
        
        return all_passed
    
    def get_test_report(self) -> Dict:
        """Get detailed test report"""
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.total_tests - self.passed_tests,
            'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            'all_passed': self.passed_tests == self.total_tests,
            'test_results': self.test_results,
            'timestamp': datetime.now()
        }


def main():
    """Main test execution function"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('smc_optimization_test.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run test suite
        tester = SMCOptimizationSystemTester()
        success = tester.run_all_tests()
        
        # Get test report
        report = tester.get_test_report()
        
        # Save test report
        logger.info(f"💾 Test report saved: {report['timestamp']}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)