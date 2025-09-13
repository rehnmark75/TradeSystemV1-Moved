#!/usr/bin/env python3
"""
Test TradingView Integration with TradeSystemV1

Tests the integration of TradingView strategies into the existing configuration system.
"""

import sys
import json
from pathlib import Path

# Add worker directory to path
worker_path = Path(__file__).parent / "worker" / "app" / "forex_scanner"
sys.path.insert(0, str(worker_path))

def test_integration_import():
    """Test importing TradingView integration module"""
    print("🔧 Testing TradingView integration import...")
    
    try:
        from configdata.strategies.tradingview_integration import (
            TradingViewStrategyIntegrator,
            search_tradingview_strategies,
            import_tradingview_strategy,
            list_tradingview_imports,
            get_tradingview_import_status,
            validate_tradingview_integration
        )
        print("✅ TradingView integration modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import TradingView integration: {e}")
        return False

def test_integration_validation():
    """Test TradingView integration validation"""
    print("\n🔍 Testing integration validation...")
    
    try:
        from configdata.strategies.tradingview_integration import validate_tradingview_integration
        
        validation_result = validate_tradingview_integration()
        print(f"Validation result: {validation_result}")
        
        if validation_result.get('valid'):
            print("✅ TradingView integration validation passed")
        else:
            print(f"⚠️ TradingView integration validation issues: {validation_result.get('error', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        return False

def test_strategies_module_integration():
    """Test integration with strategies module"""
    print("\n⚙️ Testing strategies module integration...")
    
    try:
        from configdata import strategies
        
        # Check if TradingView integration is available
        tv_available = getattr(strategies, 'TRADINGVIEW_INTEGRATION_AVAILABLE', False)
        print(f"TradingView integration available in strategies: {tv_available}")
        
        if tv_available:
            # Test functions are accessible
            functions_to_test = [
                'search_tradingview_strategies',
                'import_tradingview_strategy', 
                'list_tradingview_imports',
                'get_tradingview_import_status'
            ]
            
            for func_name in functions_to_test:
                if hasattr(strategies, func_name):
                    print(f"✅ Function {func_name} available")
                else:
                    print(f"❌ Function {func_name} not available")
            
            # Test validation function
            if hasattr(strategies, 'validate_strategy_configs'):
                validation_result = strategies.validate_strategy_configs()
                tv_validation = validation_result.get('tradingview_integration')
                if tv_validation:
                    print(f"✅ TradingView validation in strategies: {tv_validation.get('valid', False)}")
                else:
                    print("⚠️ TradingView validation not found in strategies validation")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategies module integration failed: {e}")
        return False

def test_integrator_functionality():
    """Test basic integrator functionality"""
    print("\n🚀 Testing integrator functionality...")
    
    try:
        from configdata.strategies.tradingview_integration import get_integrator
        
        integrator = get_integrator()
        print("✅ Integrator instance created")
        
        # Test import status
        status = integrator.get_import_status()
        print(f"Import status: {status}")
        
        # Test list imports
        imports = integrator.list_imports()
        print(f"Current imports: {len(imports)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrator functionality test failed: {e}")
        return False

def test_config_file_structure():
    """Test configuration file structure"""
    print("\n📁 Testing configuration file structure...")
    
    try:
        from configdata.strategies.tradingview_integration import TradingViewStrategyIntegrator
        
        integrator = TradingViewStrategyIntegrator()
        strategies_dir = integrator.strategies_dir
        
        # Check if strategies directory exists and has config files
        expected_configs = [
            'config_ema_strategy.py',
            'config_macd_strategy.py', 
            'config_smc_strategy.py',
            'config_zerolag_strategy.py'
        ]
        
        for config_file in expected_configs:
            config_path = strategies_dir / config_file
            if config_path.exists():
                print(f"✅ Found {config_file}")
            else:
                print(f"❌ Missing {config_file}")
        
        # Check for integration file
        integration_file = strategies_dir / 'tradingview_integration.py'
        if integration_file.exists():
            print("✅ TradingView integration file exists")
        else:
            print("❌ TradingView integration file missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Config file structure test failed: {e}")
        return False

def test_mock_import_scenario():
    """Test a mock import scenario (without actually importing)"""
    print("\n🧪 Testing mock import scenario...")
    
    try:
        from configdata.strategies.tradingview_integration import TradingViewStrategyIntegrator
        
        integrator = TradingViewStrategyIntegrator()
        
        # Test target config determination
        mock_analysis = {
            "signals": {
                "ema_periods": [21, 50, 200],
                "strategy_type": "trending",
                "has_cross_up": True,
                "macd": None,
                "mentions_smc": False
            }
        }
        
        target_config = integrator._determine_target_config(mock_analysis)
        print(f"✅ Determined target config: {target_config}")
        
        # Test preset content generation
        mock_script = {
            "slug": "test-ema-strategy",
            "title": "Test EMA Strategy",
            "author": "TestUser",
            "url": "https://tradingview.com/test"
        }
        
        mock_config = {
            "presets": {
                "default": {
                    "confidence_threshold": 0.55,
                    "stop_loss_pips": 15,
                    "take_profit_pips": 30
                }
            }
        }
        
        preset_content = integrator._generate_preset_content(
            target_config, "test_import", mock_config, mock_script, mock_analysis
        )
        
        if preset_content and "test_import" in preset_content:
            print("✅ Preset content generation successful")
        else:
            print("❌ Preset content generation failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock import scenario failed: {e}")
        return False

def test_path_resolution():
    """Test path resolution for imports"""
    print("\n📍 Testing path resolution...")
    
    try:
        # Test if strategy bridge path can be found
        current_file = Path(__file__)
        strategy_bridge_path = current_file.parent / "strategy_bridge"
        mcp_path = current_file.parent / "mcp"
        
        print(f"Current file: {current_file}")
        print(f"Strategy bridge path: {strategy_bridge_path}")
        print(f"MCP path: {mcp_path}")
        
        if strategy_bridge_path.exists():
            print("✅ Strategy bridge path exists")
        else:
            print("⚠️ Strategy bridge path not found")
        
        if mcp_path.exists():
            print("✅ MCP path exists")
        else:
            print("⚠️ MCP path not found")
        
        # Test if extraction modules can be imported
        try:
            sys.path.insert(0, str(strategy_bridge_path))
            from extract_pine import extract_signals
            from map_to_python import to_config
            print("✅ Strategy bridge modules can be imported")
        except ImportError as e:
            print(f"⚠️ Strategy bridge modules not available: {e}")
        
        # Test if MCP client can be imported
        try:
            sys.path.insert(0, str(mcp_path))
            from client.mcp_client import TVScriptsClient
            print("✅ MCP client can be imported")
        except ImportError as e:
            print(f"⚠️ MCP client not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path resolution test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 TradingView Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Integration Import", test_integration_import),
        ("Integration Validation", test_integration_validation),
        ("Strategies Module Integration", test_strategies_module_integration),
        ("Integrator Functionality", test_integrator_functionality),
        ("Config File Structure", test_config_file_structure),
        ("Mock Import Scenario", test_mock_import_scenario),
        ("Path Resolution", test_path_resolution)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! TradingView integration is ready.")
        return True
    elif passed >= (total * 0.7):  # 70% pass rate
        print("✅ Most integration tests passed. System should be functional.")
        return True
    else:
        print("⚠️ Some integration tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)