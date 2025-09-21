#!/usr/bin/env python3
"""
Test using the EXACT same method as the old backtest_ema.py
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
    try:
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner.core.data_fetcher import DataFetcher
        from forex_scanner.core.strategies.ema_strategy import EMAStrategy
        from forex_scanner import config
    except ImportError as e2:
        print(f"Import error: {e2}")
        sys.exit(1)

def test_exact_old_method():
    """Test using EXACT same initialization as old backtest_ema.py"""
    print("🔍 Testing EXACT OLD BACKTEST METHOD")
    print("=" * 60)

    # Initialize components EXACTLY like old backtest
    db_manager = DatabaseManager(config.DATABASE_URL)
    data_fetcher = DataFetcher(db_manager, 'UTC')  # Note: old uses UTC timezone

    # Get same data
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

    # Initialize strategy EXACTLY like old backtest (line 169)
    print(f"\n🎯 Initializing EMA Strategy (OLD BACKTEST METHOD)")
    strategy = EMAStrategy(data_fetcher=data_fetcher, backtest_mode=True)

    # Set epic like old backtest does (line 153)
    if hasattr(strategy, '_epic'):
        strategy._epic = epic
        print(f"   Set strategy._epic = {epic}")

    print(f"   backtest_mode: {strategy.backtest_mode}")
    print(f"   enhanced_validation: {strategy.enhanced_validation}")

    # Run EXACT same loop as old backtest (lines 438-458)
    print(f"\n📈 Running EXACT old backtest loop...")
    signals = []
    min_bars = getattr(config, 'MIN_BARS_FOR_SIGNAL', 50)

    print(f"   Scanning from bar {min_bars} to {len(df)}")

    for i in range(min_bars, len(df)):
        try:
            # Get data up to current point (simulate real-time)
            current_data = df.iloc[:i+1].copy()

            # Use EXACT same method as old backtest
            if strategy.enable_mtf_analysis and strategy.mtf_analyzer:
                signal = strategy.detect_signal_with_mtf(
                    current_data, epic, config.SPREAD_PIPS, timeframe
                )
            else:
                # Get current timestamp (old backtest method)
                current_timestamp = df.index[i] if i < len(df) else df.index[-1]

                signal = strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe,
                    evaluation_time=current_timestamp
                )

            if signal:
                signals.append(signal)
                print(f"   ✅ SIGNAL at bar {i}: {signal.get('signal_type', 'Unknown')} - {signal.get('confidence_score', 0):.1%}")

        except Exception as e:
            print(f"   💥 Error at bar {i}: {e}")
            if i < min_bars + 5:  # Only show first few errors
                import traceback
                traceback.print_exc()

    print(f"\n📊 RESULTS:")
    print(f"   Total bars scanned: {len(df) - min_bars}")
    print(f"   Signals found: {len(signals)}")

if __name__ == "__main__":
    test_exact_old_method()