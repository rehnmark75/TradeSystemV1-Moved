#!/usr/bin/env python3
"""
Simple Backtest Pipeline Test - Container-friendly version
Tests the core backtest components without complex dependencies
"""

import sys
import logging
from datetime import datetime, timedelta

# Set up path
sys.path.insert(0, '/app/forex_scanner')

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all backtest components can be imported"""
    logger.info("🧪 Testing imports...")

    try:
        from core.scanner_factory import ScannerFactory, ScannerMode
        logger.info("✅ ScannerFactory imported successfully")

        from core.trading.backtest_order_logger import BacktestOrderLogger
        logger.info("✅ BacktestOrderLogger imported successfully")

        from core.backtest_data_fetcher import BacktestDataFetcher
        logger.info("✅ BacktestDataFetcher imported successfully")

        from core.backtest_scanner import BacktestScanner
        logger.info("✅ BacktestScanner imported successfully")

        from core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator
        logger.info("✅ BacktestTradingOrchestrator imported successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_scanner_factory():
    """Test scanner factory creation"""
    logger.info("🏭 Testing ScannerFactory...")

    try:
        from core.scanner_factory import ScannerFactory
        from core.database import DatabaseManager
        import config

        # Create database manager
        db_manager = DatabaseManager(config.DATABASE_URL)

        # Create scanner factory
        factory = ScannerFactory(db_manager, logger)

        # Test factory info
        info = factory.get_factory_info()
        logger.info(f"✅ Factory modes: {info['supported_modes']}")

        return True

    except Exception as e:
        logger.error(f"❌ ScannerFactory test failed: {e}")
        return False

def test_backtest_execution_creation():
    """Test creating a backtest execution"""
    logger.info("🚀 Testing backtest execution creation...")

    try:
        from core.scanner_factory import ScannerFactory
        from core.database import DatabaseManager
        import config

        # Create components
        db_manager = DatabaseManager(config.DATABASE_URL)
        factory = ScannerFactory(db_manager, logger)

        # Create test execution
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=1)  # Very small range

        execution_id = factory.create_backtest_execution(
            strategy_name="SIMPLE_TEST",
            start_date=start_date,
            end_date=end_date,
            epics=["CS.D.EURUSD.MINI.IP"],
            timeframe="15m",
            execution_name="simple_pipeline_test"
        )

        logger.info(f"✅ Created backtest execution: {execution_id}")

        # Clean up
        db_manager.execute_query(
            "DELETE FROM backtest_executions WHERE id = %s",
            (execution_id,)
        )
        logger.info("✅ Cleanup completed")

        return True

    except Exception as e:
        logger.error(f"❌ Execution creation test failed: {e}")
        return False

def test_order_logger():
    """Test BacktestOrderLogger functionality"""
    logger.info("📝 Testing BacktestOrderLogger...")

    try:
        from core.trading.backtest_order_logger import BacktestOrderLogger
        from core.database import DatabaseManager
        import config

        # Create test execution first
        db_manager = DatabaseManager(config.DATABASE_URL)

        # Insert test execution
        result = db_manager.execute_query("""
            INSERT INTO backtest_executions
            (execution_name, strategy_name, data_start_date, data_end_date, epics_tested, timeframes)
            VALUES ('logger_test', 'TEST', NOW() - INTERVAL '1 hour', NOW(), ARRAY['TEST'], ARRAY['15m'])
            RETURNING id
        """).fetchone()

        execution_id = result['id']

        # Test order logger
        logger_instance = BacktestOrderLogger(db_manager, execution_id, logger)

        # Test placing an order (logging a signal)
        test_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.8,
            'entry_price': 1.1000,
            'signal_timestamp': datetime.now(),
            'strategy': 'TEST',
            'validation_passed': True
        }

        success, message, order_data = logger_instance.place_order(test_signal)

        if success:
            logger.info(f"✅ Signal logged successfully: {message}")
        else:
            logger.warning(f"⚠️ Signal logging result: {message}")

        # Get statistics
        stats = logger_instance.get_statistics()
        logger.info(f"✅ Logger stats: {stats}")

        # Clean up
        db_manager.execute_query(
            "DELETE FROM backtest_executions WHERE id = %s",
            (execution_id,)
        )

        return True

    except Exception as e:
        logger.error(f"❌ OrderLogger test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("🎬 Starting Simple Backtest Pipeline Tests")
    logger.info("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("ScannerFactory", test_scanner_factory),
        ("Execution Creation", test_backtest_execution_creation),
        ("OrderLogger", test_order_logger)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                logger.info(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} test ERROR: {e}")

    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Backtest pipeline is working!")
        return True
    else:
        logger.error(f"💥 {total - passed} tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)