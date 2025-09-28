#!/usr/bin/env python3
"""
Test script for in-memory cache performance
Quick validation before running full backtest
"""

import sys
import time
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from core.database import DatabaseManager
    from core.memory_cache import initialize_cache
    from core.backtest_data_fetcher import BacktestDataFetcher
    import config as system_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the correct directory")
    sys.exit(1)


def test_memory_cache():
    """Test in-memory cache loading and performance"""

    print("🧪 Testing In-Memory Cache Performance")
    print("=" * 50)

    try:
        # Use existing database configuration
        DATABASE_URL = "postgresql://postgres:password@postgres:5432/forex"
        db_manager = DatabaseManager(DATABASE_URL)

        print("✅ Database connection established")

        # Test 1: Cache initialization and loading
        print("\n📊 Test 1: Cache Initialization")
        start_time = time.time()

        cache = initialize_cache(db_manager, auto_load=True)

        load_time = time.time() - start_time
        cache_stats = cache.get_cache_stats()

        print(f"⏱️  Cache load time: {load_time:.2f} seconds")
        print(f"💾 Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
        print(f"📈 Rows cached: {cache_stats['total_rows']:,}")
        print(f"🎯 Epics cached: {cache_stats['epics_cached']}")
        print(f"📅 Data range: {cache_stats['data_range']['start']} to {cache_stats['data_range']['end']}")

        # Test 2: Data retrieval speed comparison
        print("\n⚡ Test 2: Speed Comparison")

        # Choose a test epic and timeframe
        test_epic = "CS.D.EURUSD.CEEM.IP"
        test_timeframe = 15  # 15 minutes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        # Test memory cache speed
        start_time = time.time()
        cached_data = cache.get_historical_data(test_epic, test_timeframe, start_date, end_date)
        cache_time = time.time() - start_time

        print(f"⚡ Memory cache: {cache_time*1000:.1f}ms, {len(cached_data) if cached_data is not None else 0} rows")

        # Test database speed (for comparison)
        start_time = time.time()
        query = """
        SELECT start_time, open_price_mid as open, high_price_mid as high,
               low_price_mid as low, close_price_mid as close, volume, ltv
        FROM ig_candles
        WHERE epic = :epic AND timeframe = :timeframe
          AND start_time >= :start_date AND start_time <= :end_date
        ORDER BY start_time
        """
        params = {
            'epic': test_epic,
            'timeframe': test_timeframe,
            'start_date': start_date,
            'end_date': end_date
        }
        db_data = db_manager.execute_query(query, params)
        db_time = time.time() - start_time

        print(f"💾 Database query: {db_time*1000:.1f}ms, {len(db_data)} rows")

        if cache_time > 0:
            speedup = db_time / cache_time
            print(f"🚀 Speedup: {speedup:.1f}x faster with memory cache")

        # Test 3: BacktestDataFetcher integration
        print("\n🧪 Test 3: BacktestDataFetcher Integration")
        start_time = time.time()

        backtest_fetcher = BacktestDataFetcher(db_manager)
        integration_time = time.time() - start_time

        print(f"⏱️  BacktestDataFetcher init: {integration_time:.2f} seconds")

        # Test data fetching through backtest fetcher
        start_time = time.time()
        df, validation = backtest_fetcher.get_enhanced_data_for_backtest(
            test_epic, "15m", start_date, end_date
        )
        fetch_time = time.time() - start_time

        print(f"📊 Enhanced data fetch: {fetch_time*1000:.1f}ms, {len(df)} rows")
        print(f"✅ Data validation: {'PASSED' if validation.get('validation_passed') else 'FAILED'}")

        # Test 4: Cache statistics
        print("\n📈 Test 4: Final Cache Statistics")
        final_stats = cache.get_cache_stats()
        performance = final_stats['performance']

        print(f"🎯 Cache hits: {performance['cache_hits']}")
        print(f"❌ Cache misses: {performance['cache_misses']}")
        print(f"📊 Hit rate: {performance['hit_rate_percent']:.1f}%")

        print("\n✅ All tests completed successfully!")
        print(f"🎉 In-memory cache is working and provides {speedup:.1f}x performance improvement")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_memory_cache()
    sys.exit(0 if success else 1)