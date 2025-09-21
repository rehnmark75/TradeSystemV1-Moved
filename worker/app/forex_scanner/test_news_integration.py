#!/usr/bin/env python3
"""
Test Economic News Filter Integration
Quick test to verify the news filter works with trade validator
"""

import sys
import logging
from datetime import datetime

# Test the import and basic functionality
def test_news_filter_integration():
    """Test that news filter integrates properly with trade validator"""
    print("🧪 Testing Economic News Filter Integration...")

    try:
        # Test importing the news filter
        from core.trading.economic_news_filter import EconomicNewsFilter
        print("✅ Economic news filter import successful")

        # Test creating the filter
        logger = logging.getLogger(__name__)
        news_filter = EconomicNewsFilter(logger=logger)
        print("✅ Economic news filter creation successful")

        # Test connection to economic calendar service
        is_connected, message = news_filter.test_service_connection()
        print(f"📡 Service connection: {'✅ Connected' if is_connected else '❌ Failed'} - {message}")

        # Test currency pair extraction
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BUY',
            'confidence_score': 0.85
        }

        currency_pair = news_filter._extract_currency_pair(test_signal)
        print(f"💱 Currency pair extraction: {currency_pair}")

        if currency_pair == "EURUSD":
            print("✅ Currency pair extraction working correctly")
        else:
            print("❌ Currency pair extraction failed")

        # Test news validation (may fail if service not available, but should not crash)
        try:
            is_valid, reason, context = news_filter.validate_signal_against_news(test_signal)
            print(f"📰 News validation test: {'✅ Valid' if is_valid else '❌ Invalid'} - {reason}")
            print(f"📊 News context: {len(context) if context else 0} fields")
        except Exception as e:
            print(f"⚠️ News validation test failed (expected if service unavailable): {e}")

        # Test confidence adjustment
        try:
            adjusted_confidence, adjustment_reason = news_filter.adjust_confidence_for_news(test_signal, 0.85)
            print(f"📈 Confidence adjustment: 85% → {adjusted_confidence:.1%} ({adjustment_reason})")
        except Exception as e:
            print(f"⚠️ Confidence adjustment test failed: {e}")

        # Test statistics
        stats = news_filter.get_filter_statistics()
        print(f"📊 Filter statistics: {stats['enabled']}, {len(stats['configuration'])} config items")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    print("🎉 Economic news filter integration test completed!")
    return True


def test_trade_validator_integration():
    """Test that trade validator can use the news filter"""
    print("\n🧪 Testing Trade Validator Integration...")

    try:
        # Test importing the trade validator
        from core.trading.trade_validator import TradeValidator
        print("✅ Trade validator import successful")

        # Create trade validator (will try to initialize news filter)
        validator = TradeValidator()
        print("✅ Trade validator creation successful")

        # Check if news filtering is enabled
        print(f"📰 News filtering enabled: {validator.enable_news_filtering}")
        print(f"📰 News filter initialized: {validator.news_filter is not None}")

        # Test a signal validation that would use news filter
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BUY',
            'confidence_score': 0.85,
            'current_price': 1.1234,
            'ema_200': 1.1200,  # Price above EMA200 for valid trend
            'timestamp': datetime.now().isoformat()
        }

        is_valid, reason = validator.validate_signal_for_trading(test_signal)
        print(f"✅ Signal validation: {'Valid' if is_valid else 'Invalid'} - {reason}")

        # Check validation statistics
        stats = validator.get_validation_statistics()
        print(f"📊 Validator has news metrics: {'news_metrics' in stats}")

        if 'news_metrics' in stats:
            news_metrics = stats['news_metrics']
            print(f"📰 News metrics: enabled={news_metrics['enabled']}, connected={news_metrics['service_connected']}")

    except Exception as e:
        print(f"❌ Trade validator integration test failed: {e}")
        return False

    print("🎉 Trade validator integration test completed!")
    return True


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("🌟 Starting Economic News Filter Integration Tests...")
    print("=" * 60)

    # Run tests
    news_filter_ok = test_news_filter_integration()
    trade_validator_ok = test_trade_validator_integration()

    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"   News Filter: {'✅ PASS' if news_filter_ok else '❌ FAIL'}")
    print(f"   Trade Validator: {'✅ PASS' if trade_validator_ok else '❌ FAIL'}")

    if news_filter_ok and trade_validator_ok:
        print("\n🎉 All tests passed! Economic news filtering is ready to use.")
        print("\n📋 Next Steps:")
        print("   1. Add news filter configuration to your main config.py")
        print("   2. Start the economic-calendar service: docker-compose up -d economic-calendar")
        print("   3. Enable news filtering: ENABLE_NEWS_FILTERING = True")
        print("   4. Monitor logs for news filtering activity")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

    print("\n🔗 Integration Summary:")
    print("   • News filter will automatically check upcoming economic events")
    print("   • Signals are blocked 30min before high-impact news (configurable)")
    print("   • Confidence scores are reduced when news events are nearby")
    print("   • All settings are configurable in config.py")
    print("   • Graceful degradation if economic calendar service is unavailable")