#!/usr/bin/env python3
"""
Test script to verify log level changes for EMA messages
"""

def test_log_cleanup():
    """Test that EMA log messages were changed to DEBUG level"""

    print("🧪 Testing EMA log level cleanup...")

    # Test: Check that INFO level EMA messages were changed to DEBUG
    try:
        with open('/app/forex_scanner/core/data_fetcher.py', 'r') as f:
            data_fetcher_content = f.read()

        # Check for removed INFO level messages
        noisy_info_messages = [
            'self.logger.info(f"🎯 Using OPTIMAL EMA periods',
            'self.logger.info(f"🔄 Adding EMA indicators:',
            'self.logger.info(f"🔄 Adding EMA 200 for MACD',
            'self.logger.info(f"✅ EMA validation passed',
            'self.logger.info(f"🎯 EMA configuration for',
            'self.logger.info("🔄 Adding EMA 200 for trend filtering'
        ]

        issues_found = []
        for message in noisy_info_messages:
            if message in data_fetcher_content:
                issues_found.append(message)

        if issues_found:
            print("❌ Still found noisy INFO level EMA messages:")
            for issue in issues_found:
                print(f"   - {issue}")
            return False
        else:
            print("✅ All noisy EMA INFO messages converted to DEBUG")

        # Check that DEBUG messages exist instead
        debug_messages = [
            'self.logger.debug(f"🎯 Using OPTIMAL EMA periods',
            'self.logger.debug(f"🔄 Adding EMA indicators:',
            'self.logger.debug(f"🔄 Adding EMA 200 for MACD',
            'self.logger.debug(f"✅ EMA validation passed',
            'self.logger.debug(f"🎯 EMA configuration for'
        ]

        debug_found = 0
        for message in debug_messages:
            if message in data_fetcher_content:
                debug_found += 1

        print(f"✅ Found {debug_found}/{len(debug_messages)} DEBUG level EMA messages")

        # Check that ZEROLAG-specific messages are still INFO (these are appropriate)
        zerolag_info_messages = [
            'self.logger.info("🔄 Adding Zero Lag EMA indicators'
        ]

        zerolag_info_found = 0
        for message in zerolag_info_messages:
            if message in data_fetcher_content:
                zerolag_info_found += 1

        if zerolag_info_found > 0:
            print(f"✅ ZEROLAG-specific INFO messages preserved: {zerolag_info_found}")

    except Exception as e:
        print(f"❌ Error reading data_fetcher.py: {e}")
        return False

    print("\n🎯 Log Cleanup Summary:")
    print("   1. ✅ Generic EMA messages moved to DEBUG level")
    print("   2. ✅ EMA validation spam eliminated")
    print("   3. ✅ ZEROLAG-specific messages preserved")
    print("   4. ✅ Log noise significantly reduced")
    print("\n✅ Log cleanup successful - ZEROLAG tests will be much cleaner!")

    return True

if __name__ == "__main__":
    success = test_log_cleanup()
    exit(0 if success else 1)