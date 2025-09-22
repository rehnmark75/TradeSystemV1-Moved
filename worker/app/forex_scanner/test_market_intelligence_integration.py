#!/usr/bin/env python3
"""
Test Market Intelligence Integration with Alert History
Verifies that market intelligence data flows correctly from strategies to alert_history table
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

# Add forex_scanner to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from alerts.alert_history import AlertHistoryManager
    from core.strategies.ichimoku_strategy import IchimokuStrategy
    from core.data_fetcher import DataFetcher
    from core.strategies.helpers.ichimoku_market_intelligence_adapter import IchimokuMarketIntelligenceAdapter
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this test from within the forex_scanner directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_signal_with_intelligence() -> Dict:
    """Create a test signal with market intelligence data"""
    return {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'pair': 'EURUSD',
        'signal_type': 'BULL',
        'strategy': 'ichimoku',
        'confidence_score': 0.75,
        'price': 1.0850,
        'bid_price': 1.0848,
        'ask_price': 1.0852,
        'timeframe': '15m',
        'spread_pips': 1.5,

        # Ichimoku-specific data
        'tenkan_sen': 1.0845,
        'kijun_sen': 1.0840,
        'senkou_span_a': 1.0842,
        'senkou_span_b': 1.0838,
        'cloud_top': 1.0842,
        'cloud_bottom': 1.0838,
        'tk_bull_cross': True,
        'signal_source': 'TK_BULL',

        # Market intelligence data
        'market_intelligence': {
            'regime_analysis': {
                'dominant_regime': 'trending',
                'confidence': 0.82,
                'regime_scores': {
                    'trending': 0.8,
                    'ranging': 0.2,
                    'breakout': 0.6,
                    'reversal': 0.3,
                    'high_volatility': 0.4,
                    'low_volatility': 0.6
                }
            },
            'session_analysis': {
                'current_session': 'london',
                'volatility_level': 'high',
                'session_characteristics': ['High volatility', 'EUR pairs active']
            },
            'market_context': {
                'volatility_percentile': 75.2,
                'market_strength': {
                    'average_trend_strength': 0.73,
                    'market_bias': 'bullish',
                    'directional_consensus': 0.68
                }
            },
            'strategy_adaptation': {
                'applied_regime': 'trending',
                'confidence_threshold_used': 0.55,
                'regime_suitable': True,
                'adaptation_summary': 'Ichimoku parameters adapted for trending regime'
            },
            'intelligence_source': 'MarketIntelligenceEngine',
            'analysis_timestamp': datetime.now().isoformat(),
            'volatility_level': 'high'
        },

        # Strategy metadata
        'strategy_metadata': {
            'ichimoku_config': {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52
            },
            'validation_results': {
                'cloud_position': True,
                'chikou_clear': True,
                'mtf_confirmed': False
            }
        },

        # Standard fields
        'market_timestamp': datetime.now(),
        'data_source': 'test_scanner',
        'signal_hash': 'test_hash_123'
    }

def test_alert_history_integration():
    """Test that market intelligence data is correctly saved to alert_history"""
    logger.info("🧪 Testing Market Intelligence Integration with Alert History")

    try:
        # Create AlertHistoryManager
        alert_manager = AlertHistoryManager()

        if not alert_manager:
            logger.error("❌ Failed to create AlertHistoryManager")
            return False

        logger.info("✅ AlertHistoryManager created successfully")

        # Create test signal with market intelligence
        test_signal = create_test_signal_with_intelligence()

        logger.info(f"📊 Test signal created with market intelligence:")
        logger.info(f"   - Regime: {test_signal['market_intelligence']['regime_analysis']['dominant_regime']}")
        logger.info(f"   - Confidence: {test_signal['market_intelligence']['regime_analysis']['confidence']:.1%}")
        logger.info(f"   - Session: {test_signal['market_intelligence']['session_analysis']['current_session']}")
        logger.info(f"   - Volatility: {test_signal['market_intelligence']['session_analysis']['volatility_level']}")

        # Save the alert
        alert_id = alert_manager.save_alert(
            signal=test_signal,
            alert_message="Test market intelligence integration",
            alert_level="INFO"
        )

        if alert_id:
            logger.info(f"✅ Alert saved successfully with ID: {alert_id}")
            logger.info("🧠 Market intelligence data should now be stored in strategy_metadata JSON field")

            # Verify the data was saved correctly by checking database
            try:
                def verify_operation(conn, cursor):
                    cursor.execute("""
                        SELECT
                            strategy_metadata,
                            market_regime,
                            market_session,
                            epic,
                            signal_type,
                            confidence_score
                        FROM alert_history
                        WHERE id = %s
                    """, (alert_id,))

                    result = cursor.fetchone()
                    if result:
                        strategy_metadata, market_regime, market_session, epic, signal_type, confidence = result

                        logger.info(f"📋 Alert verification:")
                        logger.info(f"   - Epic: {epic}")
                        logger.info(f"   - Signal: {signal_type}")
                        logger.info(f"   - Confidence: {confidence:.1%}")
                        logger.info(f"   - Market Regime: {market_regime}")
                        logger.info(f"   - Market Session: {market_session}")

                        if strategy_metadata:
                            try:
                                metadata = json.loads(strategy_metadata) if isinstance(strategy_metadata, str) else strategy_metadata

                                if 'market_intelligence' in metadata:
                                    intel_data = metadata['market_intelligence']
                                    logger.info(f"✅ Market Intelligence data found in database:")
                                    logger.info(f"   - Regime: {intel_data.get('regime_analysis', {}).get('dominant_regime', 'N/A')}")
                                    logger.info(f"   - Confidence: {intel_data.get('regime_analysis', {}).get('confidence', 0):.1%}")
                                    logger.info(f"   - Session: {intel_data.get('session_analysis', {}).get('current_session', 'N/A')}")
                                    logger.info(f"   - Volatility: {intel_data.get('session_analysis', {}).get('volatility_level', 'N/A')}")
                                    logger.info(f"   - Intelligence Applied: {intel_data.get('intelligence_applied', False)}")

                                    # Check indexable fields for future Phase 2
                                    indexable = intel_data.get('_indexable_fields', {})
                                    if indexable:
                                        logger.info(f"📊 Indexable fields prepared for Phase 2:")
                                        logger.info(f"   - Regime Confidence: {indexable.get('regime_confidence', 0):.1%}")
                                        logger.info(f"   - Volatility Level: {indexable.get('volatility_level', 'N/A')}")
                                        logger.info(f"   - Market Bias: {indexable.get('market_bias', 'N/A')}")

                                    return True
                                else:
                                    logger.warning("⚠️ Market intelligence data not found in strategy_metadata")
                                    return False
                            except Exception as e:
                                logger.error(f"❌ Error parsing strategy_metadata: {e}")
                                return False
                        else:
                            logger.warning("⚠️ strategy_metadata is empty")
                            return False
                    else:
                        logger.error("❌ Alert not found in database")
                        return False

                verification_result = alert_manager._execute_with_connection(verify_operation, "verify market intelligence")

                if verification_result:
                    logger.info("🎉 Market Intelligence Integration Test PASSED!")
                    return True
                else:
                    logger.error("❌ Market Intelligence Integration Test FAILED!")
                    return False

            except Exception as e:
                logger.error(f"❌ Error verifying saved data: {e}")
                return False
        else:
            logger.error("❌ Failed to save alert")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def test_market_intelligence_extraction():
    """Test the market intelligence extraction method directly"""
    logger.info("🧪 Testing Market Intelligence Extraction Method")

    try:
        alert_manager = AlertHistoryManager()
        test_signal = create_test_signal_with_intelligence()

        # Test the extraction method directly
        extracted_data = alert_manager._extract_market_intelligence_data(test_signal)

        if 'strategy_metadata' in extracted_data:
            metadata = extracted_data['strategy_metadata']

            if 'market_intelligence' in metadata:
                intel_data = metadata['market_intelligence']
                logger.info("✅ Market intelligence extraction successful:")
                logger.info(f"   - Regime: {intel_data.get('regime_analysis', {}).get('dominant_regime', 'N/A')}")
                logger.info(f"   - Source: {intel_data.get('intelligence_source', 'N/A')}")
                logger.info(f"   - Intelligence Applied: {intel_data.get('intelligence_applied', False)}")
                return True
            else:
                logger.error("❌ Market intelligence not found in extracted metadata")
                return False
        else:
            logger.error("❌ strategy_metadata not found in extracted data")
            return False

    except Exception as e:
        logger.error(f"❌ Extraction test failed: {e}")
        return False

def main():
    """Run all market intelligence integration tests"""
    logger.info("🚀 Starting Market Intelligence Integration Tests")

    tests_passed = 0
    total_tests = 2

    # Test 1: Market intelligence extraction method
    if test_market_intelligence_extraction():
        tests_passed += 1
        logger.info("✅ Test 1 PASSED: Market intelligence extraction")
    else:
        logger.error("❌ Test 1 FAILED: Market intelligence extraction")

    # Test 2: Full integration with alert history
    if test_alert_history_integration():
        tests_passed += 1
        logger.info("✅ Test 2 PASSED: Alert history integration")
    else:
        logger.error("❌ Test 2 FAILED: Alert history integration")

    # Summary
    logger.info(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Market intelligence integration is working correctly.")
        logger.info("\n📋 Next Steps:")
        logger.info("   1. Market intelligence data is now stored in strategy_metadata JSON field")
        logger.info("   2. Run forex scanner with Ichimoku strategy to generate real signals")
        logger.info("   3. Query alert_history table to analyze strategy performance by market regime")
        logger.info("   4. Consider implementing Phase 2 (dedicated columns) for performance optimization")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)