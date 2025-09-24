#!/usr/bin/env python3
"""
Basic RAG Intelligence Strategy Integration Test
==============================================

Simple test to verify the RAG Intelligence Strategy can be imported
and initialized without external dependencies.
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")

    try:
        # Test core strategy import
        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy
        print("✅ RAGIntelligenceStrategy imported successfully")

        # Test configuration import
        from configdata.strategies.config_rag_intelligence_strategy import RAGIntelligenceConfig
        print("✅ RAGIntelligenceConfig imported successfully")

        # Test helper imports
        from core.strategies.helpers.market_intelligence_analyzer import MarketIntelligenceAnalyzer
        print("✅ MarketIntelligenceAnalyzer imported successfully")

        from core.strategies.helpers.rag_integration_helper import RAGIntegrationHelper
        print("✅ RAGIntegrationHelper imported successfully")

        # Test strategy is in __init__.py
        from core.strategies import RAGIntelligenceStrategy as StrategyFromInit
        print("✅ RAGIntelligenceStrategy available in strategies module")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_initialization():
    """Test basic strategy initialization"""
    print("\n🔧 Testing basic initialization...")

    try:
        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy

        # Test basic initialization without external dependencies
        strategy = RAGIntelligenceStrategy(
            epic="CS.D.EURUSD.MINI.IP",
            data_fetcher=None,  # No database required for basic test
            backtest_mode=True
        )

        # Check basic attributes
        assert strategy.name == 'rag_intelligence'
        assert strategy.epic == "CS.D.EURUSD.MINI.IP"
        assert strategy.backtest_mode is True

        print("✅ Basic initialization successful")
        print(f"   Strategy name: {strategy.name}")
        print(f"   Epic: {strategy.epic}")
        print(f"   Backtest mode: {strategy.backtest_mode}")

        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")

    try:
        from configdata.strategies.config_rag_intelligence_strategy import RAGIntelligenceConfig

        config = RAGIntelligenceConfig()

        # Check key configuration attributes
        assert hasattr(config, 'STRATEGY_NAME')
        assert hasattr(config, 'MIN_CONFIDENCE')
        assert hasattr(config, 'MARKET_REGIMES')
        assert hasattr(config, 'RAG_QUERY_TEMPLATES')

        print("✅ Configuration loaded successfully")
        print(f"   Strategy name: {config.STRATEGY_NAME}")
        print(f"   Min confidence: {config.MIN_CONFIDENCE}")
        print(f"   Available regimes: {list(config.MARKET_REGIMES.keys())}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_backtest_import():
    """Test backtest module import"""
    print("\n🧪 Testing backtest import...")

    try:
        # Add backtest directory to path
        backtest_dir = os.path.join(script_dir, 'backtests')
        sys.path.insert(0, backtest_dir)

        # Import would require pandas, so just check file exists
        backtest_file = os.path.join(backtest_dir, 'backtest_rag_intelligence.py')
        if os.path.exists(backtest_file):
            print("✅ Backtest file exists")

            # Check if file has main components by reading it
            with open(backtest_file, 'r') as f:
                content = f.read()
                if 'class RAGIntelligenceBacktest' in content:
                    print("✅ RAGIntelligenceBacktest class found")
                if 'def run_backtest' in content:
                    print("✅ run_backtest method found")
                if 'def main()' in content:
                    print("✅ main function found")

            return True
        else:
            print("❌ Backtest file not found")
            return False

    except Exception as e:
        print(f"❌ Backtest test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🧪 RAG Intelligence Strategy Basic Integration Test")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_basic_initialization),
        ("Configuration Test", test_configuration),
        ("Backtest Import Test", test_backtest_import)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"⚠️ {test_name} failed")

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ RAG Intelligence Strategy is properly integrated")
        print("\n📋 Next Steps:")
        print("   1. Test in Docker environment with database")
        print("   2. Run backtest: python backtest_rag_intelligence.py --epic CS.D.EURUSD.MINI.IP --days 1")
        print("   3. Integrate with main forex scanner")
        return 0
    else:
        print("❌ Some tests failed. Please fix before proceeding.")
        return 1

if __name__ == "__main__":
    exit(main())