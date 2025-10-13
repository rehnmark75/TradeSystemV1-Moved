#!/usr/bin/env python3
"""
Debug Volume Profile Backtest - Check why no signals are generated
"""

print("🔍 Volume Profile Backtest Diagnostics")
print("=" * 60)

# Step 1: Check if strategy is enabled in config
print("\n1️⃣ Checking config.VOLUME_PROFILE_STRATEGY...")
try:
    import config
    vp_enabled = getattr(config, 'VOLUME_PROFILE_STRATEGY', False)
    print(f"   {'✅' if vp_enabled else '❌'} VOLUME_PROFILE_STRATEGY = {vp_enabled}")
    if not vp_enabled:
        print("   ⚠️  Strategy is DISABLED in config.py!")
        print("   💡 Set VOLUME_PROFILE_STRATEGY = True in config.py")
except Exception as e:
    print(f"   ❌ Error loading config: {e}")

# Step 2: Check SignalDetector initialization
print("\n2️⃣ Checking SignalDetector initialization...")
try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector

    db_manager = DatabaseManager(config.DATABASE_URL)
    detector = SignalDetector(db_manager)

    if hasattr(detector, 'volume_profile_strategy'):
        if detector.volume_profile_strategy is not None:
            print(f"   ✅ Volume Profile strategy initialized: {type(detector.volume_profile_strategy).__name__}")
        else:
            print(f"   ❌ volume_profile_strategy is None (strategy disabled in config)")
    else:
        print(f"   ❌ volume_profile_strategy attribute doesn't exist")

except Exception as e:
    print(f"   ❌ Error initializing SignalDetector: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Check if detect_volume_profile_signals method exists
print("\n3️⃣ Checking detect_volume_profile_signals method...")
try:
    if hasattr(detector, 'detect_volume_profile_signals'):
        print(f"   ✅ detect_volume_profile_signals method exists")

        # Try calling it with dummy data
        print("   🧪 Testing method call...")
        result = detector.detect_volume_profile_signals(
            epic='CS.D.EURUSD.CEEM.IP',
            pair='EURUSD',
            spread_pips=1.5,
            timeframe='15m'
        )
        print(f"   📊 Method returned: {type(result)} = {result}")
    else:
        print(f"   ❌ detect_volume_profile_signals method doesn't exist")
except Exception as e:
    print(f"   ❌ Error calling method: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Check backtest_scanner strategy mapping
print("\n4️⃣ Checking backtest_scanner strategy mapping...")
try:
    from core.backtest_scanner import BacktestScanner

    # Check if VP is in the mapping
    test_config = {
        'strategy_name': 'VOLUME_PROFILE',
        'start_date': None,
        'end_date': None,
        'epics': ['CS.D.EURUSD.CEEM.IP'],
        'timeframe': '15m',
        'pipeline_mode': False
    }

    print(f"   ✅ BacktestScanner imported successfully")
    print(f"   💡 Strategy name will be: 'VOLUME_PROFILE' (uppercase)")

except Exception as e:
    print(f"   ❌ Error with BacktestScanner: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Check Volume Profile imports
print("\n5️⃣ Checking Volume Profile component imports...")
try:
    from analysis.volume_profile import VolumeProfile, VolumeNode
    print(f"   ✅ VolumeProfile data structures imported")
except Exception as e:
    print(f"   ❌ Error importing VolumeProfile: {e}")

try:
    from core.strategies.helpers.volume_profile_calculator import VolumeProfileCalculator
    print(f"   ✅ VolumeProfileCalculator imported")
except Exception as e:
    print(f"   ❌ Error importing VolumeProfileCalculator: {e}")

try:
    from core.strategies.helpers.volume_profile_analyzer import VolumeProfileAnalyzer
    print(f"   ✅ VolumeProfileAnalyzer imported")
except Exception as e:
    print(f"   ❌ Error importing VolumeProfileAnalyzer: {e}")

try:
    from core.strategies.volume_profile_strategy import VolumeProfileStrategy
    print(f"   ✅ VolumeProfileStrategy imported")

    # Try instantiating it
    vp_strategy = VolumeProfileStrategy()
    print(f"   ✅ VolumeProfileStrategy instantiated successfully")
    print(f"   📊 Min confidence: {vp_strategy.min_confidence}")
    print(f"   📊 Lookback periods: {vp_strategy.lookback_periods}")

except Exception as e:
    print(f"   ❌ Error with VolumeProfileStrategy: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🔍 Diagnostic complete!")
print("\n💡 Common issues:")
print("   1. VOLUME_PROFILE_STRATEGY = False in config.py")
print("   2. Import errors in Volume Profile files")
print("   3. SignalDetector not initializing volume_profile_strategy")
print("   4. Insufficient data (need 70+ bars)")
print("\n🚀 If all checks pass, try:")
print("   python bt.py EURUSD 7 VP --show-signals")
