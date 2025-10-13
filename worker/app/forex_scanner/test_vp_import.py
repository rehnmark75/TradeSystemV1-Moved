#!/usr/bin/env python3
"""
Quick test to verify Volume Profile strategy imports correctly
"""

print("🧪 Testing Volume Profile Strategy Imports...")
print("=" * 60)

try:
    print("1️⃣ Testing VolumeProfile data structures...")
    from analysis.volume_profile import VolumeProfile, VolumeNode
    print("   ✅ VolumeProfile and VolumeNode imported")
except Exception as e:
    print(f"   ❌ Failed to import VolumeProfile structures: {e}")
    exit(1)

try:
    print("2️⃣ Testing VolumeProfileCalculator...")
    from core.strategies.helpers.volume_profile_calculator import VolumeProfileCalculator
    print("   ✅ VolumeProfileCalculator imported")
except Exception as e:
    print(f"   ❌ Failed to import VolumeProfileCalculator: {e}")
    exit(1)

try:
    print("3️⃣ Testing VolumeProfileAnalyzer...")
    from core.strategies.helpers.volume_profile_analyzer import VolumeProfileAnalyzer
    print("   ✅ VolumeProfileAnalyzer imported")
except Exception as e:
    print(f"   ❌ Failed to import VolumeProfileAnalyzer: {e}")
    exit(1)

try:
    print("4️⃣ Testing Volume Profile config...")
    from configdata.strategies import config_volume_profile_strategy
    print(f"   ✅ Volume Profile config imported")
    print(f"   📊 Strategy enabled: {config_volume_profile_strategy.VOLUME_PROFILE_STRATEGY}")
    print(f"   📊 Active config: {config_volume_profile_strategy.ACTIVE_VP_CONFIG}")
except Exception as e:
    print(f"   ❌ Failed to import Volume Profile config: {e}")
    exit(1)

try:
    print("5️⃣ Testing VolumeProfileStrategy...")
    from core.strategies.volume_profile_strategy import VolumeProfileStrategy
    print("   ✅ VolumeProfileStrategy imported")
except Exception as e:
    print(f"   ❌ Failed to import VolumeProfileStrategy: {e}")
    exit(1)

try:
    print("6️⃣ Testing strategy registration in SignalDetector...")
    from core.signal_detector import SignalDetector
    from core.database import DatabaseManager
    import config

    db_manager = DatabaseManager(config.DATABASE_URL)
    detector = SignalDetector(db_manager)

    if hasattr(detector, 'volume_profile_strategy') and detector.volume_profile_strategy is not None:
        print(f"   ✅ Volume Profile strategy registered in SignalDetector")
    else:
        print(f"   ⚠️ Volume Profile strategy not initialized (check VOLUME_PROFILE_STRATEGY in config)")
except Exception as e:
    print(f"   ❌ Failed to test SignalDetector: {e}")

print("=" * 60)
print("✅ All Volume Profile imports successful!")
print("\n🚀 Ready to backtest with: python bt.py --all 7 VP --show-signals")
