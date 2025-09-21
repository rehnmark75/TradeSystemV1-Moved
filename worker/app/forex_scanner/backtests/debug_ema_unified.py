#!/usr/bin/env python3
"""
Debug EMA strategy using UNIFIED BACKTEST parameters
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

def debug_ema_unified():
    """Debug EMA strategy using UNIFIED BACKTEST parameters"""
    print("🔍 Debugging EMA Strategy - UNIFIED BACKTEST VERSION")
    print("=" * 60)

    # Initialize components (same as unified backtest)
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager)

    # Get same data as unified backtest
    epic = 'CS.D.EURUSD.CEEM.IP'
    timeframe = '15m'
    lookback_hours = 30 * 24  # 30 days

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

    # Initialize EMA strategy with UNIFIED BACKTEST parameters
    print(f"\n🎯 Initializing EMA Strategy (UNIFIED BACKTEST MODE)")
    strategy = EMAStrategy(
        backtest_mode=True,
        epic=epic,
        use_optimal_parameters=True
    )

    print(f"   backtest_mode: {strategy.backtest_mode}")
    print(f"   enhanced_validation: {strategy.enhanced_validation}")
    print(f"   ema_config: {strategy.ema_config}")
    print(f"   min_confidence: {strategy.min_confidence}")

    # Test signal detection at several points (same as unified backtest loop)
    min_bars = 50
    test_points = [100, 200, 300, 400, 500, len(df) - 1]

    print(f"\n📈 Testing signal detection...")
    signals_found = 0

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
                signals_found += 1
                print(f"   ✅ SIGNAL FOUND: {signal.get('signal_type', 'Unknown')} - {signal.get('confidence_score', 0):.2%}")
                print(f"      Reason: {signal.get('reason', 'No reason')}")
            else:
                print("   ❌ No signal")

        except Exception as e:
            print(f"   💥 Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n📊 SUMMARY:")
    print(f"   Total test points: {len(test_points)}")
    print(f"   Signals found: {signals_found}")

if __name__ == "__main__":
    debug_ema_unified()