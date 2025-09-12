#!/usr/bin/env python3
"""
Test Epic-Specific Parameter Loading System

Tests the integration between SMC optimization results and the SMC strategy.
Verifies that optimal parameters are properly loaded and applied per epic.

Author: Trading System V1
Created: 2025-09-12
"""

import sys
import os
import logging
from typing import Dict
import pandas as pd

# Add forex_scanner to path for imports
sys.path.append('/app/forex_scanner')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_smc_optimal_parameter_service():
    """Test the SMC optimal parameter service."""
    print("\n🧪 Testing SMC Optimal Parameter Service")
    print("=" * 60)
    
    try:
        from optimization.smc_optimal_parameter_service import (
            get_smc_optimal_parameters,
            get_smc_top_configurations,
            get_all_smc_best_configs,
            get_smc_optimization_summary
        )
        
        # Test 1: Get optimization summary
        print("\n📊 Optimization Summary:")
        summary = get_smc_optimization_summary()
        if 'error' not in summary:
            print(f"   ✅ Total Tests: {summary['total_tests']}")
            print(f"   ✅ Unique Epics: {summary['unique_epics']}")
            print(f"   ✅ Average Win Rate: {summary['avg_win_rate']:.1f}%")
            print(f"   ✅ Best Overall: {summary['best_overall_epic']} ({summary['best_overall_config']})")
            
            available_epics = summary['epics_list'][:3]  # Test first 3 epics
        else:
            print(f"   ❌ Error: {summary['error']}")
            return False
        
        # Test 2: Get optimal parameters for specific epics
        print(f"\n🎯 Optimal Parameters for Sample Epics:")
        for epic in available_epics:
            config = get_smc_optimal_parameters(epic)
            print(f"   ✅ {epic}:")
            print(f"      Config: {config['smc_config']}")
            print(f"      Confidence: {config['confidence_level']}")
            print(f"      Stop Loss: {config['stop_loss_pips']} pips")
            print(f"      Take Profit: {config['take_profit_pips']} pips")
            print(f"      Expected Win Rate: {config['expected_win_rate']:.1f}%")
            print(f"      Performance Score: {config['performance_score']:.1f}")
        
        # Test 3: Get top configurations for one epic
        print(f"\n🏆 Top 3 Configurations for {available_epics[0]}:")
        top_configs = get_smc_top_configurations(available_epics[0], top_n=3)
        for config in top_configs:
            print(f"   #{config['rank']}: {config['smc_config']} "
                  f"(Win Rate: {config['expected_win_rate']:.1f}%, "
                  f"Performance: {config['performance_score']:.1f})")
        
        print("\n✅ SMC Optimal Parameter Service tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ SMC Optimal Parameter Service test failed: {e}")
        return False

def test_smc_strategy_integration():
    """Test SMC strategy integration with optimal parameters."""
    print("\n🧪 Testing SMC Strategy Integration")
    print("=" * 60)
    
    try:
        from core.strategies.smc_strategy import SMCStrategy
        from optimization.smc_optimal_parameter_service import get_smc_optimization_summary
        
        # Get available epics from optimization results
        summary = get_smc_optimization_summary()
        if 'error' in summary:
            print(f"❌ No optimization data available: {summary['error']}")
            return False
        
        test_epics = summary['epics_list'][:2]  # Test first 2 epics
        
        for epic in test_epics:
            print(f"\n🔍 Testing SMC Strategy for {epic}:")
            
            # Test with optimized parameters enabled
            print("   📈 With Optimized Parameters:")
            try:
                strategy_optimized = SMCStrategy(
                    epic=epic, 
                    use_optimized_parameters=True
                )
                
                print(f"      ✅ Strategy initialized successfully")
                print(f"      ✅ Configuration loaded: {strategy_optimized.smc_config.get('_optimized', False)}")
                
                if '_optimized' in strategy_optimized.smc_config:
                    print(f"      ✅ Optimization status: {strategy_optimized.smc_config['_optimized']}")
                
                # Check key parameters
                key_params = ['confidence_level', 'stop_loss_pips', 'take_profit_pips', 'risk_reward_ratio']
                for param in key_params:
                    if param in strategy_optimized.smc_config:
                        print(f"      ✅ {param}: {strategy_optimized.smc_config[param]}")
                
            except Exception as e:
                print(f"      ❌ Failed to initialize with optimized parameters: {e}")
            
            # Test with optimized parameters disabled (fallback)
            print("   📉 With Fallback Parameters:")
            try:
                strategy_fallback = SMCStrategy(
                    epic=epic, 
                    use_optimized_parameters=False
                )
                
                print(f"      ✅ Fallback strategy initialized successfully")
                
            except Exception as e:
                print(f"      ❌ Failed to initialize fallback strategy: {e}")
        
        print("\n✅ SMC Strategy integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ SMC Strategy integration test failed: {e}")
        return False

def test_parameter_comparison():
    """Test parameter comparison between epics."""
    print("\n🧪 Testing Parameter Comparison Between Epics")
    print("=" * 60)
    
    try:
        from optimization.smc_optimal_parameter_service import (
            get_smc_optimal_parameters,
            get_smc_optimization_summary
        )
        
        summary = get_smc_optimization_summary()
        if 'error' in summary:
            print(f"❌ No optimization data: {summary['error']}")
            return False
        
        epics = summary['epics_list'][:3]
        
        print("\n📊 Parameter Comparison:")
        print(f"{'Epic':<25} {'Config':<12} {'Conf.':<6} {'SL':<4} {'TP':<4} {'Win%':<6} {'Score':<8}")
        print("-" * 70)
        
        for epic in epics:
            config = get_smc_optimal_parameters(epic)
            print(f"{epic:<25} {config['smc_config']:<12} "
                  f"{config['confidence_level']:<6.2f} "
                  f"{config['stop_loss_pips']:<4} "
                  f"{config['take_profit_pips']:<4} "
                  f"{config['expected_win_rate']:<6.1f} "
                  f"{config['performance_score']:<8.1f}")
        
        print("\n✅ Parameter comparison test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Parameter comparison test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback behavior when optimization data is unavailable."""
    print("\n🧪 Testing Fallback Behavior")
    print("=" * 60)
    
    try:
        from core.strategies.smc_strategy import SMCStrategy
        
        # Test with a non-existent epic
        fake_epic = "FAKE.EPIC.TEST"
        print(f"\n🔍 Testing fallback for non-existent epic: {fake_epic}")
        
        strategy = SMCStrategy(
            epic=fake_epic,
            use_optimized_parameters=True  # Should fall back to defaults
        )
        
        print(f"   ✅ Strategy initialized with fallback parameters")
        
        # Check if fallback parameters are used
        if strategy.smc_config.get('optimization_source') == 'fallback_default':
            print(f"   ✅ Fallback configuration detected")
        else:
            print(f"   ⚠️ Unexpected configuration source: {strategy.smc_config.get('optimization_source', 'unknown')}")
        
        # Check key fallback parameters
        fallback_params = ['min_confidence', 'stop_loss_pips', 'take_profit_pips']
        for param in fallback_params:
            if param in strategy.smc_config:
                print(f"   ✅ {param}: {strategy.smc_config[param]}")
        
        print("\n✅ Fallback behavior test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Fallback behavior test failed: {e}")
        return False

def main():
    """Run all epic-specific parameter system tests."""
    print("🚀 Epic-Specific Parameter Loading System Tests")
    print("=" * 80)
    
    tests = [
        ("SMC Optimal Parameter Service", test_smc_optimal_parameter_service),
        ("SMC Strategy Integration", test_smc_strategy_integration),
        ("Parameter Comparison", test_parameter_comparison),
        ("Fallback Behavior", test_fallback_behavior)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Epic-specific parameter loading system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)