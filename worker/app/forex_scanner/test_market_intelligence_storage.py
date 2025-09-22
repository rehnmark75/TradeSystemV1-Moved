#!/usr/bin/env python3
"""
Test Market Intelligence Storage Implementation

Tests the complete market intelligence storage system including:
- Database schema creation
- MarketIntelligenceHistoryManager functionality
- Scanner integration
- Analytics queries

Run this script to verify the implementation works correctly.
"""

import sys
import os
import logging
from datetime import datetime, timezone
import json

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_schema():
    """Test database schema creation"""
    logger.info("🧪 Testing database schema creation...")

    try:
        from core.database import DatabaseManager
        import config

        # Initialize database manager with config URL
        db_manager = DatabaseManager(config.DATABASE_URL)
        logger.info("✅ DatabaseManager initialized")

        # Test schema creation by importing the history manager
        from core.intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager

        # This will automatically create the table
        history_manager = MarketIntelligenceHistoryManager(db_manager)
        logger.info("✅ Database schema created successfully")

        return True, history_manager, db_manager

    except Exception as e:
        logger.error(f"❌ Database schema test failed: {e}")
        return False, None, None

def test_market_intelligence_engine():
    """Test market intelligence engine functionality"""
    logger.info("🧪 Testing market intelligence engine...")

    try:
        from core.database import DatabaseManager
        from core.data_fetcher import DataFetcher
        from core.intelligence.market_intelligence import MarketIntelligenceEngine
        import config

        db_manager = DatabaseManager(config.DATABASE_URL)
        data_fetcher = DataFetcher(db_manager)
        engine = MarketIntelligenceEngine(data_fetcher)

        # Test with a small epic list
        test_epics = ['CS.D.EURUSD.TODAY.IP', 'CS.D.GBPUSD.TODAY.IP']

        # Generate test intelligence report
        report = engine.generate_market_intelligence_report(test_epics)

        if report and 'market_regime' in report:
            logger.info("✅ Market intelligence engine working")

            # Log key report details
            market_regime = report.get('market_regime', {})
            dominant_regime = market_regime.get('dominant_regime', 'unknown')
            confidence = market_regime.get('confidence', 0.5)

            session_analysis = report.get('session_analysis', {})
            current_session = session_analysis.get('current_session', 'unknown')

            logger.info(f"   📊 Generated report: {dominant_regime} regime ({confidence:.1%}) during {current_session} session")

            return True, report
        else:
            logger.warning("⚠️ Market intelligence engine returned empty/invalid report")
            return False, None

    except Exception as e:
        logger.error(f"❌ Market intelligence engine test failed: {e}")
        return False, None

def test_intelligence_storage(history_manager, db_manager):
    """Test market intelligence storage functionality"""
    logger.info("🧪 Testing market intelligence storage...")

    try:
        # Create sample intelligence report
        sample_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'market_regime': {
                'dominant_regime': 'trending',
                'confidence': 0.75,
                'regime_scores': {
                    'trending': 0.75,
                    'ranging': 0.25,
                    'breakout': 0.4,
                    'reversal': 0.3,
                    'high_volatility': 0.6,
                    'low_volatility': 0.4
                },
                'market_strength': {
                    'average_trend_strength': 0.7,
                    'average_volatility': 0.6,
                    'market_bias': 'bullish',
                    'directional_consensus': 0.8,
                    'market_efficiency': 0.65
                },
                'correlation_analysis': {
                    'correlation_matrix': {},
                    'currency_strength': {'USD': 0.3, 'EUR': -0.2},
                    'risk_on_off': 'risk_on'
                },
                'recommended_strategy': {
                    'strategy': 'trend_following',
                    'ema_config': 'aggressive',
                    'recommendations': ['Use trend-following strategies', 'Look for pullback entries']
                }
            },
            'session_analysis': {
                'current_session': 'london',
                'session_config': {
                    'volatility': 'high',
                    'characteristics': 'High volatility, strong trend potential',
                    'preferred_pairs': ['EURUSD', 'GBPUSD'],
                    'strategy_adjustment': 'Trend following strategies',
                    'risk_level': 'medium'
                },
                'optimal_timeframes': ['5m', '15m']
            },
            'trading_recommendations': {
                'primary_strategy': 'trend_following',
                'confidence_threshold': 0.7,
                'position_sizing': 'NORMAL',
                'timeframe_focus': ['15m']
            },
            'confidence_score': 0.75
        }

        test_epics = ['CS.D.EURUSD.TODAY.IP', 'CS.D.GBPUSD.TODAY.IP']
        scan_cycle_id = f"test_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Test saving intelligence data
        record_id = history_manager.save_market_intelligence(
            intelligence_report=sample_report,
            epic_list=test_epics,
            scan_cycle_id=scan_cycle_id
        )

        if record_id:
            logger.info(f"✅ Market intelligence stored successfully - Record ID: {record_id}")

            # Test retrieval
            recent_data = history_manager.get_recent_intelligence(hours=1, limit=5)

            if not recent_data.empty:
                logger.info(f"✅ Retrieved {len(recent_data)} recent intelligence records")

                # Display sample data
                latest = recent_data.iloc[0]
                logger.info(f"   📊 Latest: {latest['dominant_regime']} regime ({latest['regime_confidence']:.1%}) "
                          f"in {latest['current_session']} session")

                return True, record_id
            else:
                logger.warning("⚠️ No data retrieved after storage")
                return False, None
        else:
            logger.error("❌ Failed to store market intelligence")
            return False, None

    except Exception as e:
        logger.error(f"❌ Intelligence storage test failed: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False, None

def test_analytics_functionality(db_manager):
    """Test analytics functionality"""
    logger.info("🧪 Testing analytics functionality...")

    try:
        from core.intelligence.market_intelligence_analytics import MarketIntelligenceAnalytics

        analytics = MarketIntelligenceAnalytics(db_manager)

        # Test dashboard generation
        dashboard = analytics.get_market_intelligence_summary_dashboard(hours=24)

        if dashboard and 'summary_statistics' in dashboard:
            logger.info("✅ Analytics dashboard generated successfully")

            stats = dashboard.get('summary_statistics', {})
            total_scans = stats.get('total_scans', 0)
            logger.info(f"   📊 Dashboard shows {total_scans} scans in last 24 hours")

            return True
        else:
            logger.warning("⚠️ Analytics dashboard generation failed or returned empty data")
            return False

    except Exception as e:
        logger.error(f"❌ Analytics test failed: {e}")
        return False

def test_scanner_integration():
    """Test scanner integration (mock test)"""
    logger.info("🧪 Testing scanner integration...")

    try:
        # Test that scanner can import the new modules
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        import config

        # Create scanner instance (this tests import and initialization)
        db_manager = DatabaseManager(config.DATABASE_URL)
        scanner = IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=['CS.D.EURUSD.TODAY.IP'],
            min_confidence=0.7,
            scan_interval=60
        )

        # Check if market intelligence components are initialized
        has_intelligence = hasattr(scanner, 'enable_market_intelligence') and scanner.enable_market_intelligence
        has_engine = hasattr(scanner, 'market_intelligence_engine') and scanner.market_intelligence_engine is not None
        has_history = hasattr(scanner, 'market_intelligence_history') and scanner.market_intelligence_history is not None

        if has_intelligence and has_engine and has_history:
            logger.info("✅ Scanner integration successful")
            logger.info(f"   🧠 Market Intelligence: {'✅' if has_intelligence else '❌'}")
            logger.info(f"   🔍 Intelligence Engine: {'✅' if has_engine else '❌'}")
            logger.info(f"   💾 Intelligence History: {'✅' if has_history else '❌'}")
            return True
        else:
            logger.warning("⚠️ Scanner integration incomplete")
            logger.info(f"   🧠 Market Intelligence: {'✅' if has_intelligence else '❌'}")
            logger.info(f"   🔍 Intelligence Engine: {'✅' if has_engine else '❌'}")
            logger.info(f"   💾 Intelligence History: {'✅' if has_history else '❌'}")
            return False

    except Exception as e:
        logger.error(f"❌ Scanner integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of the market intelligence storage system"""
    logger.info("🚀 Starting comprehensive market intelligence storage test...")
    logger.info("=" * 80)

    test_results = {}

    # Test 1: Database Schema
    logger.info("\n1️⃣ Database Schema Test")
    schema_success, history_manager, db_manager = test_database_schema()
    test_results['database_schema'] = schema_success

    if not schema_success:
        logger.error("❌ Cannot continue without database schema")
        return test_results

    # Test 2: Market Intelligence Engine
    logger.info("\n2️⃣ Market Intelligence Engine Test")
    engine_success, sample_report = test_market_intelligence_engine()
    test_results['intelligence_engine'] = engine_success

    # Test 3: Intelligence Storage
    logger.info("\n3️⃣ Intelligence Storage Test")
    storage_success, record_id = test_intelligence_storage(history_manager, db_manager)
    test_results['intelligence_storage'] = storage_success

    # Test 4: Analytics
    logger.info("\n4️⃣ Analytics Test")
    analytics_success = test_analytics_functionality(db_manager)
    test_results['analytics'] = analytics_success

    # Test 5: Scanner Integration
    logger.info("\n5️⃣ Scanner Integration Test")
    scanner_success = test_scanner_integration()
    test_results['scanner_integration'] = scanner_success

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 80)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")

    logger.info(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Market Intelligence Storage is ready for use.")
    elif passed_tests >= total_tests * 0.8:
        logger.info("⚠️ Most tests passed. Minor issues may need attention.")
    else:
        logger.error("❌ Multiple test failures. System needs debugging.")

    return test_results

if __name__ == "__main__":
    """Run the test when script is executed directly"""
    try:
        results = run_comprehensive_test()

        # Exit with appropriate code
        total_tests = len(results)
        passed_tests = sum(results.values())

        if passed_tests == total_tests:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)