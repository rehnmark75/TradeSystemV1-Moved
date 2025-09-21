#!/usr/bin/env python3
"""
Debug EMA strategy to see why no signals are generated
"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
forex_scanner_dir = os.path.dirname(script_dir)
app_dir = os.path.dirname(forex_scanner_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, forex_scanner_dir)
sys.path.insert(0, app_dir)

try:
    from core.database import DatabaseManager
    from core.data_fetcher import DataFetcher
    from core.strategies.ema_strategy import EMAStrategy
    import config
except ImportError as e:
    print(f"Import error: {e}")
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.strategies.ema_strategy import EMAStrategy
        from forex_scanner import config
    except ImportError as e2:
        print(f"Fallback import error: {e2}")
        sys.exit(1)

def debug_ema_strategy():
    """Debug EMA strategy signal generation"""
    print("🔍 Debugging EMA Strategy")
    print("=" * 50)

    # Initialize components
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager)

    # Get some data
    epic = 'CS.D.EURUSD.CEEM.IP'
    timeframe = '15m'
    lookback_hours = 14 * 24  # 14 days

    print(f"📊 Fetching data for {epic} ({timeframe})")
    df = data_fetcher.get_enhanced_data(
        epic=epic,
        pair='EURUSD',
        timeframe=timeframe,
        lookback_hours=lookback_hours
    )

    if df is None or len(df) == 0:
        print("❌ No data fetched")
        return

    print(f"✅ Data fetched: {len(df)} bars")
    print(f"   Latest: {df.index[-1]}")
    print(f"   Earliest: {df.index[0]}")

    # Initialize EMA strategy
    print("\n🎯 Initializing EMA Strategy")
    strategy = EMAStrategy()

    # Test signal detection at several points
    min_bars = 50
    test_points = [100, 200, 300, 400, 500, len(df) - 1]

    for i in test_points:
        if i >= len(df):
            continue

        print(f"\n📈 Testing at bar {i} ({df.index[i]})")
        current_data = df.iloc[:i+1].copy()

        try:
            signal = strategy.detect_signal(
                current_data, epic, config.SPREAD_PIPS, timeframe
            )

            if signal:
                print(f"   ✅ SIGNAL FOUND: {signal.get('signal_type', 'Unknown')} - {signal.get('confidence_score', 0):.2%}")
                print(f"      Reason: {signal.get('reason', 'No reason')}")
            else:
                print("   ❌ No signal")

        except Exception as e:
            print(f"   💥 Error: {e}")
            import traceback
            traceback.print_exc()

    # Check recent EMA values
    print(f"\n📊 Recent EMA values:")
    if 'ema_5' in df.columns and 'ema_13' in df.columns and 'ema_50' in df.columns:
        for i in range(max(0, len(df) - 5), len(df)):
            print(f"   {df.index[i]}: EMA5={df.iloc[i]['ema_5']:.5f}, EMA13={df.iloc[i]['ema_13']:.5f}, EMA50={df.iloc[i]['ema_50']:.5f}")
    else:
        print("   ❌ EMA columns not found")
        print(f"   Available columns: {df.columns.tolist()}")

if __name__ == "__main__":
    debug_ema_strategy()