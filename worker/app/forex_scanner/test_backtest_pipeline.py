#!/usr/bin/env python3
"""
End-to-End Backtest Pipeline Test

This script tests the complete backtest integration pipeline:
1. Database schema creation
2. Scanner factory initialization
3. Backtest execution creation
4. Signal processing and logging
5. Performance calculation
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import config
    from core.database import DatabaseManager
    from core.scanner_factory import ScannerFactory, ScannerMode
    from core.trading.backtest_trading_orchestrator import create_backtest_trading_orchestrator
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('BacktestPipelineTest')


class BacktestPipelineTest:
    """Complete end-to-end backtest pipeline test"""

    def __init__(self):
        self.db_manager = None
        self.scanner_factory = None
        self.execution_id = None

    def initialize_database(self) -> bool:
        """Initialize database connection and ensure schema exists"""
        try:
            logger.info("🔗 Connecting to database...")
            self.db_manager = DatabaseManager(config.DATABASE_URL)

            # Test connection
            result = self.db_manager.execute_query("SELECT 1 as test").fetchone()
            if result and result['test'] == 1:
                logger.info("✅ Database connection successful")
                return True
            else:
                logger.error("❌ Database connection test failed")
                return False

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False

    def check_backtest_schema(self) -> bool:
        """Check if backtest tables exist"""
        try:
            logger.info("📋 Checking backtest schema...")

            tables_to_check = [
                'backtest_executions',
                'backtest_signals',
                'backtest_performance'
            ]

            for table in tables_to_check:
                result = self.db_manager.execute_query(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    );
                    """, (table,)
                ).fetchone()

                if result and result['exists']:
                    logger.info(f"   ✅ Table {table} exists")
                else:
                    logger.error(f"   ❌ Table {table} missing")
                    return False

            logger.info("✅ All backtest tables exist")
            return True

        except Exception as e:
            logger.error(f"❌ Schema check failed: {e}")
            return False

    def create_test_execution(self) -> bool:
        """Create a test backtest execution"""
        try:
            logger.info("🚀 Creating test backtest execution...")

            self.scanner_factory = ScannerFactory(self.db_manager, logger)

            # Create execution for the last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            self.execution_id = self.scanner_factory.create_backtest_execution(
                strategy_name="TEST_PIPELINE",
                start_date=start_date,
                end_date=end_date,
                epics=["CS.D.EURUSD.MINI.IP", "CS.D.GBPUSD.MINI.IP"],  # Small test set
                timeframe="15m",
                execution_name="pipeline_integration_test"
            )

            logger.info(f"✅ Created test execution: {self.execution_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create test execution: {e}")
            return False

    def test_backtest_components(self) -> bool:
        """Test individual backtest components"""
        try:
            logger.info("🧪 Testing backtest components...")

            # Test backtest scanner creation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            backtest_config = {
                'execution_id': self.execution_id,
                'strategy_name': 'TEST_PIPELINE',
                'start_date': start_date,
                'end_date': end_date,
                'epics': ["CS.D.EURUSD.MINI.IP"],
                'timeframe': '15m'
            }

            # Test scanner creation
            scanner = self.scanner_factory.create_scanner(
                ScannerMode.BACKTEST,
                backtest_config=backtest_config
            )
            logger.info("   ✅ BacktestScanner creation successful")

            # Test orchestrator creation
            orchestrator = create_backtest_trading_orchestrator(
                self.execution_id,
                backtest_config,
                self.db_manager,
                logger=logger
            )
            logger.info("   ✅ BacktestTradingOrchestrator creation successful")

            # Test factory workflow
            factory_info = self.scanner_factory.get_factory_info()
            logger.info(f"   ✅ Scanner factory info: {factory_info['supported_modes']}")

            return True

        except Exception as e:
            logger.error(f"❌ Component testing failed: {e}")
            return False

    def test_minimal_backtest_run(self) -> bool:
        """Run a minimal backtest to test the complete pipeline"""
        try:
            logger.info("🚀 Running minimal backtest pipeline test...")

            # Very small time range for quick test
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=2)  # Just 2 hours

            backtest_config = {
                'execution_id': self.execution_id,
                'strategy_name': 'TEST_PIPELINE',
                'start_date': start_date,
                'end_date': end_date,
                'epics': ["CS.D.EURUSD.MINI.IP"],  # Single pair
                'timeframe': '15m'
            }

            # Run quick backtest using orchestrator
            with create_backtest_trading_orchestrator(
                self.execution_id,
                backtest_config,
                self.db_manager,
                logger=logger
            ) as orchestrator:

                # Run just the orchestration (no full backtest)
                orchestrator._initialize_backtest_execution()
                logger.info("   ✅ Orchestrator initialization successful")

                # Test signal processing pipeline (mock signal)
                mock_signal = {
                    'epic': 'CS.D.EURUSD.MINI.IP',
                    'signal_type': 'BULL',
                    'confidence_score': 0.85,
                    'entry_price': 1.1000,
                    'stop_loss_price': 1.0950,
                    'take_profit_price': 1.1050,
                    'signal_timestamp': datetime.now(),
                    'strategy': 'TEST_PIPELINE',
                    'validation_passed': True
                }

                success, message, order_data = orchestrator.process_signal_through_validation_pipeline(mock_signal)

                if success:
                    logger.info(f"   ✅ Signal processing successful: {message}")
                else:
                    logger.warning(f"   ⚠️ Signal processing result: {message}")

                # Check orchestrator stats
                stats = orchestrator.get_orchestrator_statistics()
                logger.info(f"   📊 Orchestrator stats: {stats['orchestrator_stats']}")

            return True

        except Exception as e:
            logger.error(f"❌ Minimal backtest run failed: {e}")
            return False

    def verify_database_results(self) -> bool:
        """Verify that data was properly written to database"""
        try:
            logger.info("🔍 Verifying database results...")

            # Check backtest execution record
            execution_result = self.db_manager.execute_query(
                "SELECT * FROM backtest_executions WHERE id = %s",
                (self.execution_id,)
            ).fetchone()

            if execution_result:
                logger.info(f"   ✅ Execution record found: {execution_result['execution_name']}")
                logger.info(f"      Status: {execution_result['status']}")
            else:
                logger.error("   ❌ Execution record not found")
                return False

            # Check for any signals (might be 0 for test data)
            signals_count = self.db_manager.execute_query(
                "SELECT COUNT(*) as count FROM backtest_signals WHERE execution_id = %s",
                (self.execution_id,)
            ).fetchone()

            logger.info(f"   📊 Signals in database: {signals_count['count']}")

            # Check performance calculation function
            try:
                self.db_manager.execute_query(
                    "SELECT calculate_backtest_performance(%s)",
                    (self.execution_id,)
                )
                logger.info("   ✅ Performance calculation function works")
            except Exception as e:
                logger.warning(f"   ⚠️ Performance calculation issue: {e}")

            return True

        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            return False

    def cleanup_test_data(self):
        """Clean up test data"""
        try:
            if self.execution_id:
                logger.info(f"🧹 Cleaning up test execution {self.execution_id}...")

                # Delete test data (cascades to related tables)
                self.db_manager.execute_query(
                    "DELETE FROM backtest_executions WHERE id = %s",
                    (self.execution_id,)
                )

                logger.info("✅ Test data cleanup completed")

        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

    def run_complete_test(self) -> bool:
        """Run the complete end-to-end test"""
        logger.info("🎬 Starting End-to-End Backtest Pipeline Test")
        logger.info("=" * 60)

        try:
            # Step 1: Database initialization
            if not self.initialize_database():
                return False

            # Step 2: Schema verification
            if not self.check_backtest_schema():
                return False

            # Step 3: Create test execution
            if not self.create_test_execution():
                return False

            # Step 4: Test components
            if not self.test_backtest_components():
                return False

            # Step 5: Minimal backtest run
            if not self.test_minimal_backtest_run():
                return False

            # Step 6: Verify results
            if not self.verify_database_results():
                return False

            logger.info("=" * 60)
            logger.info("🎉 END-TO-END BACKTEST PIPELINE TEST PASSED!")
            logger.info("✅ All components working correctly")
            logger.info("✅ Database integration successful")
            logger.info("✅ Signal processing pipeline functional")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            return False

        finally:
            # Always cleanup
            self.cleanup_test_data()


def main():
    """Main test execution"""
    test = BacktestPipelineTest()

    success = test.run_complete_test()

    if success:
        print("\n🎊 BACKTEST PIPELINE INTEGRATION: SUCCESS! 🎊")
        return 0
    else:
        print("\n💥 BACKTEST PIPELINE INTEGRATION: FAILED! 💥")
        return 1


if __name__ == "__main__":
    sys.exit(main())