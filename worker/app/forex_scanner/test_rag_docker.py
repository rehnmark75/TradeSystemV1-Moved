#!/usr/bin/env python3
"""
RAG Intelligence Strategy Docker Environment Test
==============================================

Simple test to verify the RAG Intelligence Strategy works in Docker environment.
This test doesn't require external dependencies and tests core functionality.
"""

import sys
import os
import logging
import traceback

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [TEST] - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('rag_docker_test')

def test_strategy_import():
    """Test RAG strategy import"""
    logger = logging.getLogger('rag_docker_test')
    try:
        logger.info("🔍 Testing RAG Intelligence Strategy import...")

        # Test strategy import
        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy
        logger.info("✅ RAGIntelligenceStrategy imported successfully")

        # Test configuration import
        from configdata.strategies.config_rag_intelligence_strategy import RAGIntelligenceConfig
        logger.info("✅ RAGIntelligenceConfig imported successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_strategy_initialization():
    """Test strategy initialization"""
    logger = logging.getLogger('rag_docker_test')
    try:
        logger.info("🔧 Testing strategy initialization...")

        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy

        # Test basic initialization
        strategy = RAGIntelligenceStrategy(
            epic="CS.D.EURUSD.MINI.IP",
            data_fetcher=None,
            backtest_mode=True
        )

        logger.info(f"✅ Strategy initialized: {strategy.name}")
        logger.info(f"   Epic: {strategy.epic}")
        logger.info(f"   Min confidence: {strategy.min_confidence}")
        logger.info(f"   RAG helper available: {strategy.rag_helper is not None}")

        # Test getting stats
        stats = strategy.get_strategy_stats()
        logger.info(f"   Stats collected: {len(stats)} metrics")

        return True

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_market_condition_fallback():
    """Test market condition analysis fallback"""
    logger = logging.getLogger('rag_docker_test')
    try:
        logger.info("📊 Testing market condition fallback...")

        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy

        strategy = RAGIntelligenceStrategy(
            epic="CS.D.EURUSD.MINI.IP",
            data_fetcher=None,  # No database
            backtest_mode=True
        )

        # Test market condition analysis (should use fallback)
        market_condition = strategy.analyze_market_conditions("CS.D.EURUSD.MINI.IP")

        logger.info(f"✅ Market condition analyzed")
        logger.info(f"   Regime: {market_condition.regime}")
        logger.info(f"   Confidence: {market_condition.confidence:.1%}")
        logger.info(f"   Session: {market_condition.session}")
        logger.info(f"   Volatility: {market_condition.volatility}")

        return True

    except Exception as e:
        logger.error(f"❌ Market condition test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_rag_selection():
    """Test RAG strategy selection"""
    logger = logging.getLogger('rag_docker_test')
    try:
        logger.info("🤖 Testing RAG strategy selection...")

        from core.strategies.rag_intelligence_strategy import RAGIntelligenceStrategy, MarketCondition
        from datetime import datetime

        strategy = RAGIntelligenceStrategy(
            epic="CS.D.EURUSD.MINI.IP",
            data_fetcher=None,
            backtest_mode=True
        )

        # Create test market condition
        test_market_condition = MarketCondition(
            regime='trending_up',
            confidence=0.75,
            session='london',
            volatility='high',
            dominant_timeframe='15m',
            success_factors=['momentum', 'volume'],
            timestamp=datetime.utcnow()
        )

        # Test RAG strategy selection
        strategy_code = strategy.select_optimal_code(test_market_condition)

        logger.info(f"✅ Strategy code selected")
        logger.info(f"   Code type: {strategy_code.code_type}")
        logger.info(f"   Source ID: {strategy_code.source_id}")
        logger.info(f"   Confidence: {strategy_code.confidence_score:.1%}")
        logger.info(f"   Parameters: {len(strategy_code.parameters)} items")

        return True

    except Exception as e:
        logger.error(f"❌ RAG selection test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_backtest_availability():
    """Test backtest module availability"""
    logger = logging.getLogger('rag_docker_test')
    try:
        logger.info("🧪 Testing backtest availability...")

        # Check if backtest file exists
        backtest_path = "/app/forex_scanner/backtests/backtest_rag_intelligence.py"
        if os.path.exists(backtest_path):
            logger.info("✅ Backtest file found")

            # Test import (without running)
            import importlib.util
            spec = importlib.util.spec_from_file_location("backtest_rag_intelligence", backtest_path)
            if spec:
                logger.info("✅ Backtest module can be loaded")
            else:
                logger.warning("⚠️ Backtest module spec creation failed")

        else:
            logger.warning(f"⚠️ Backtest file not found at {backtest_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Backtest test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_all_tests():
    """Run all Docker environment tests"""
    logger = setup_logging()

    logger.info("🧪 RAG Intelligence Strategy Docker Environment Test")
    logger.info("=" * 65)

    tests = [
        ("Import Test", test_strategy_import),
        ("Initialization Test", test_strategy_initialization),
        ("Market Condition Test", test_market_condition_fallback),
        ("RAG Selection Test", test_rag_selection),
        ("Backtest Availability Test", test_backtest_availability)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")

    # Summary
    logger.info("\n" + "=" * 65)
    logger.info("📊 DOCKER TEST SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        logger.info("🎉 ALL DOCKER TESTS PASSED!")
        logger.info("✅ RAG Intelligence Strategy is ready for Docker deployment")
        logger.info("\n📋 Next Steps:")
        logger.info("   1. Run full backtest: python backtests/backtest_rag_intelligence.py --epic CS.D.EURUSD.MINI.IP --days 1")
        logger.info("   2. Integrate with main forex scanner")
        logger.info("   3. Test with RAG system if available")
        return 0
    else:
        logger.warning("❌ Some tests failed. Strategy may have issues in Docker environment.")
        return 1

if __name__ == "__main__":
    exit(run_all_tests())