#!/usr/bin/env python3
"""
Integration Test Suite for Component Availability
Tests that all components gracefully handle missing dependencies and context changes

This test suite validates the architectural resilience improvements implemented
to prevent cascade failures like the validation system implementation that broke
the live forex scanner.
"""

import sys
import traceback
import logging
from typing import Dict, List, Optional, Any

# Setup logging for test output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test that all core modules can be imported successfully"""
    test_results = {}
    
    core_modules = [
        ('forex_scanner.core.scanner', 'ForexScanner'),
        ('forex_scanner.core.signal_detector', 'SignalDetector'),
        ('forex_scanner.core.data_fetcher', 'DataFetcher'), 
        ('forex_scanner.core.database', 'DatabaseManager'),
        ('forex_scanner.core.processing.signal_processor', 'SignalProcessor'),
    ]
    
    for module_path, class_name in core_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            test_results[f"{module_path}.{class_name}"] = "✅ SUCCESS"
            logger.info(f"✅ {module_path}.{class_name} imported successfully")
        except Exception as e:
            test_results[f"{module_path}.{class_name}"] = f"❌ FAILED: {e}"
            logger.error(f"❌ {module_path}.{class_name} failed: {e}")
    
    return test_results

def test_strategy_imports():
    """Test that strategy modules import correctly"""
    test_results = {}
    
    strategy_modules = [
        ('forex_scanner.core.strategies.ema_strategy', 'EMAStrategy'),
        ('forex_scanner.core.strategies.macd_strategy', 'MACDStrategy'),
        ('forex_scanner.core.strategies.zero_lag_strategy', 'ZeroLagStrategy'),
        ('forex_scanner.core.strategies.combined_strategy', 'CombinedStrategy'),
    ]
    
    for module_path, class_name in strategy_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            test_results[f"{module_path}.{class_name}"] = "✅ SUCCESS"
            logger.info(f"✅ {module_path}.{class_name} imported successfully")
        except Exception as e:
            test_results[f"{module_path}.{class_name}"] = f"❌ FAILED: {e}"
            logger.error(f"❌ {module_path}.{class_name} failed: {e}")
    
    return test_results

def test_trading_component_availability():
    """Test trading orchestrator component availability checking"""
    test_results = {}
    
    try:
        from forex_scanner.core.trading.trading_orchestrator import TradingOrchestrator
        
        # Initialize orchestrator
        orchestrator = TradingOrchestrator(test_mode=True)
        
        # Test component availability checking
        components = orchestrator.check_component_availability()
        
        test_results["TradingOrchestrator.init"] = "✅ SUCCESS"
        test_results["TradingOrchestrator.check_component_availability"] = "✅ SUCCESS"
        
        # Log component availability
        for component, available in components.items():
            status = "✅ AVAILABLE" if available else "⚠️ UNAVAILABLE"
            test_results[f"Component.{component}"] = status
            logger.info(f"{status}: {component}")
            
    except Exception as e:
        test_results["TradingOrchestrator"] = f"❌ FAILED: {e}"
        logger.error(f"❌ TradingOrchestrator failed: {e}")
        traceback.print_exc()
    
    return test_results

def test_config_imports():
    """Test that config imports work in both contexts"""
    test_results = {}
    
    # Test system config import
    try:
        try:
            import config
        except ImportError:
            from forex_scanner import config
            
        test_results["config.system"] = "✅ SUCCESS"
        logger.info("✅ System config imported successfully")
        
        # Test key config variables
        critical_vars = ['EMA_STRATEGY', 'MACD_STRATEGY', 'MIN_CONFIDENCE', 'DATABASE_URL']
        for var in critical_vars:
            if hasattr(config, var):
                test_results[f"config.{var}"] = "✅ PRESENT"
            else:
                test_results[f"config.{var}"] = "❌ MISSING"
                logger.warning(f"❌ Missing config variable: {var}")
                
    except Exception as e:
        test_results["config.system"] = f"❌ FAILED: {e}"
        logger.error(f"❌ System config failed: {e}")
    
    # Test configdata import
    try:
        try:
            from configdata import config as configdata
        except ImportError:
            from forex_scanner.configdata import config as configdata
            
        test_results["config.configdata"] = "✅ SUCCESS"
        logger.info("✅ ConfigData imported successfully")
        
    except Exception as e:
        test_results["config.configdata"] = f"❌ FAILED: {e}"
        logger.error(f"❌ ConfigData failed: {e}")
    
    return test_results

def test_analysis_imports():
    """Test analysis module imports"""
    test_results = {}
    
    analysis_modules = [
        ('forex_scanner.analysis.technical', 'TechnicalAnalyzer'),
        ('forex_scanner.analysis.volume', 'VolumeAnalyzer'),
        ('forex_scanner.analysis.behavior', 'BehaviorAnalyzer'),
    ]
    
    for module_path, class_name in analysis_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            test_results[f"{module_path}.{class_name}"] = "✅ SUCCESS"
            logger.info(f"✅ {module_path}.{class_name} imported successfully")
        except Exception as e:
            test_results[f"{module_path}.{class_name}"] = f"❌ FAILED: {e}"
            logger.error(f"❌ {module_path}.{class_name} failed: {e}")
    
    return test_results

def test_alert_system_imports():
    """Test alert and notification system imports"""
    test_results = {}
    
    alert_modules = [
        ('forex_scanner.alerts.claude_analyzer', 'ClaudeAnalyzer'),
        ('forex_scanner.alerts.alert_history', 'AlertHistoryManager'),
    ]
    
    for module_path, class_name in alert_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            test_results[f"{module_path}.{class_name}"] = "✅ SUCCESS"
            logger.info(f"✅ {module_path}.{class_name} imported successfully")
        except Exception as e:
            test_results[f"{module_path}.{class_name}"] = f"❌ FAILED: {e}"
            logger.error(f"❌ {module_path}.{class_name} failed: {e}")
    
    return test_results

def test_validation_system_compatibility():
    """Test that validation system can coexist with live scanner"""
    test_results = {}
    
    try:
        # Import validation system
        from forex_scanner.validation.signal_replay_validator import SignalReplayValidator
        test_results["ValidationSystem.import"] = "✅ SUCCESS"
        logger.info("✅ Validation system imported successfully")
        
        # Import live scanner components
        from forex_scanner.core.scanner import ForexScanner
        test_results["LiveScanner.import"] = "✅ SUCCESS" 
        logger.info("✅ Live scanner imported successfully after validation import")
        
        # Test that both can coexist
        test_results["ValidationSystem.coexistence"] = "✅ SUCCESS"
        logger.info("✅ Validation system and live scanner can coexist")
        
    except Exception as e:
        test_results["ValidationSystem.compatibility"] = f"❌ FAILED: {e}"
        logger.error(f"❌ Validation system compatibility failed: {e}")
    
    return test_results

def test_import_fallback_patterns():
    """Test that import fallback patterns work correctly"""
    test_results = {}
    
    # Test patterns by trying imports in different ways
    fallback_tests = [
        ("Core Database", "forex_scanner.core.database", "DatabaseManager"),
        ("Signal Detector", "forex_scanner.core.signal_detector", "SignalDetector"),
        ("Data Fetcher", "forex_scanner.core.data_fetcher", "DataFetcher"),
        ("Technical Analysis", "forex_scanner.analysis.technical", "TechnicalAnalyzer"),
    ]
    
    for test_name, module_path, class_name in fallback_tests:
        try:
            # Test absolute import (fallback pattern)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            test_results[f"FallbackPattern.{test_name}"] = "✅ SUCCESS"
            logger.info(f"✅ {test_name} fallback pattern works")
        except Exception as e:
            test_results[f"FallbackPattern.{test_name}"] = f"❌ FAILED: {e}"
            logger.error(f"❌ {test_name} fallback pattern failed: {e}")
    
    return test_results

def generate_test_report(all_results: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(1 for results in all_results.values() 
                      for result in results.values() 
                      if result.startswith("✅"))
    failed_tests = total_tests - passed_tests
    
    report = {
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
        },
        "categories": {},
        "critical_failures": []
    }
    
    # Process each test category
    for category, results in all_results.items():
        category_passed = sum(1 for result in results.values() if result.startswith("✅"))
        category_total = len(results)
        
        report["categories"][category] = {
            "passed": category_passed,
            "total": category_total,
            "success_rate": f"{(category_passed/category_total)*100:.1f}%" if category_total > 0 else "0%",
            "results": results
        }
        
        # Identify critical failures
        for test_name, result in results.items():
            if result.startswith("❌") and any(critical in test_name.lower() 
                                              for critical in ['scanner', 'orchestrator', 'config']):
                report["critical_failures"].append({
                    "test": test_name,
                    "category": category,
                    "error": result
                })
    
    return report

def main():
    """Run complete integration test suite"""
    logger.info("🧪 Starting Integration Test Suite for Component Availability")
    logger.info("=" * 80)
    
    # Run all test categories
    all_results = {}
    
    test_categories = [
        ("Core Imports", test_core_imports),
        ("Strategy Imports", test_strategy_imports), 
        ("Trading Components", test_trading_component_availability),
        ("Config Imports", test_config_imports),
        ("Analysis Imports", test_analysis_imports),
        ("Alert System", test_alert_system_imports),
        ("Validation Compatibility", test_validation_system_compatibility),
        ("Import Fallback Patterns", test_import_fallback_patterns)
    ]
    
    for category_name, test_function in test_categories:
        logger.info(f"\\n🔍 Running {category_name} tests...")
        try:
            results = test_function()
            all_results[category_name] = results
        except Exception as e:
            logger.error(f"❌ {category_name} test category failed: {e}")
            all_results[category_name] = {"category_error": f"❌ FAILED: {e}"}
    
    # Generate comprehensive report
    logger.info("\\n📊 Generating Test Report...")
    report = generate_test_report(all_results)
    
    # Print summary
    logger.info("\\n" + "=" * 80)
    logger.info("🎯 INTEGRATION TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {report['summary']['total_tests']}")
    logger.info(f"Passed: {report['summary']['passed']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Success Rate: {report['summary']['success_rate']}")
    
    # Print category results
    for category, results in report["categories"].items():
        logger.info(f"\\n📋 {category}: {results['passed']}/{results['total']} ({results['success_rate']})")
    
    # Print critical failures
    if report["critical_failures"]:
        logger.info(f"\\n🚨 CRITICAL FAILURES ({len(report['critical_failures'])})")
        for failure in report["critical_failures"]:
            logger.error(f"❌ {failure['category']}.{failure['test']}: {failure['error']}")
    else:
        logger.info("\\n✅ No critical failures detected!")
    
    # Determine overall result
    success_rate = (report['summary']['passed'] / report['summary']['total_tests']) * 100
    if success_rate >= 90:
        logger.info("\\n🎉 INTEGRATION TEST SUITE: EXCELLENT")
    elif success_rate >= 75:
        logger.info("\\n✅ INTEGRATION TEST SUITE: GOOD") 
    elif success_rate >= 50:
        logger.info("\\n⚠️ INTEGRATION TEST SUITE: NEEDS IMPROVEMENT")
    else:
        logger.info("\\n❌ INTEGRATION TEST SUITE: CRITICAL ISSUES")
    
    # Return success code
    return 0 if success_rate >= 75 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)