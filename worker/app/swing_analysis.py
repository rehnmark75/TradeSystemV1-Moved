#!/usr/bin/env python3
"""
Swing Level Analysis for Latest 20 Trades
Analyzes if trades were taken near swing highs/lows
"""

import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import sys

# Database connection
DB_CONFIG = {
    'host': 'postgres',
    'database': 'forex',
    'user': 'postgres',
    'password': 'postgres'
}

# Trade data structure
trades_data = [
    (1697, 'CS.D.GBPUSD.MINI.IP', 'SELL', 1.341865, 1.34286, 1.340065, '2026-01-08 19:02:02.145071', -92.73, 'LOSS'),
    (1696, 'CS.D.NZDUSD.MINI.IP', 'SELL', 0.57455, 0.57545, 0.57305, '2026-01-08 18:17:08.269156', -83.36, 'LOSS'),
    (1695, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.387565, 1.38657, 1.389365, '2026-01-08 17:32:35.594055', -66.86, 'LOSS'),
    (1694, 'CS.D.AUDJPY.MINI.IP', 'BUY', 105.17150000000001, 105.0515, 105.3715, '2026-01-08 15:17:11.439387', -70.86, 'LOSS'),
    (1691, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.387765, 1.38677, 1.389565, '2026-01-08 08:05:31.350453', -66.70, 'LOSS'),
    (1690, 'CS.D.USDJPY.MINI.IP', 'SELL', 156.4795, 156.5995, 156.2795, '2026-01-08 08:05:30.920783', -70.97, 'LOSS'),
    (1687, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.385035, 1.38404, 1.386835, '2026-01-07 20:12:22.146999', 118.65, 'WIN'),
    (1685, 'CS.D.AUDUSD.MINI.IP', 'BUY', 0.67445, 0.67355, 0.67595, '2026-01-07 14:08:53.955278', -82.99, 'LOSS'),
    (1679, 'CS.D.AUDJPY.MINI.IP', 'BUY', 104.824, 104.734, 104.974, '2026-01-02 18:59:29.524421', 87.73, 'WIN'),
    (1678, 'CS.D.EURJPY.MINI.IP', 'BUY', 183.9435, 183.8535, 184.0935, '2026-01-02 18:28:49.293682', -53.19, 'LOSS'),
    (1677, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.374805, 1.37391, 1.376305, '2026-01-02 18:13:24.379462', -60.75, 'LOSS'),
    (1676, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.373645, 1.37275, 1.375145, '2026-01-02 13:07:28.062924', -60.61, 'LOSS'),
    (1675, 'CS.D.USDJPY.MINI.IP', 'BUY', 157.0045, 156.8845, 157.2045, '2026-01-02 12:40:33.769509', -70.86, 'LOSS'),
    (1674, 'CS.D.GBPUSD.MINI.IP', 'SELL', 1.344375, 1.34537, 1.342575, '2026-01-02 12:27:33.606763', -92.64, 'LOSS'),
    (1673, 'CS.D.USDCAD.MINI.IP', 'BUY', 1.371795, 1.3709, 1.373295, '2026-01-02 09:21:23.363149', 99.96, 'WIN'),
    (1672, 'CS.D.GBPUSD.MINI.IP', 'SELL', 1.3458750000000002, 1.34648, 1.345275, '2026-01-02 07:52:52.838577', -55.55, 'LOSS'),
    (1671, 'CS.D.USDJPY.MINI.IP', 'BUY', 156.89950000000002, 156.8395, None, '2026-01-02 07:42:46.719491', -35.43, 'LOSS'),
    (1670, 'CS.D.USDJPY.MINI.IP', 'BUY', 156.763, 156.703, 156.823, '2025-12-31 13:31:31.23526', 35.07, 'WIN'),
    (1668, 'CS.D.USDJPY.MINI.IP', 'SELL', 155.92, 155.98, 155.86, '2025-12-30 12:09:18.236029', -35.53, 'LOSS'),
    (1667, 'CS.D.USDCHF.MINI.IP', 'SELL', 0.7887550000000001, 0.78936, 0.788155, '2025-12-30 11:41:35.803535', -70.25, 'LOSS'),
]

def get_pip_multiplier(epic: str) -> int:
    """Get pip multiplier based on currency pair"""
    if 'JPY' in epic:
        return 100  # 2 decimal places for JPY pairs
    return 10000  # 4 decimal places for other pairs

def calculate_pips(price_diff: float, epic: str) -> float:
    """Convert price difference to pips"""
    multiplier = get_pip_multiplier(epic)
    return abs(price_diff * multiplier)

def detect_swings(candles: List[Tuple], window: int = 5) -> Tuple[List, List]:
    """
    Detect swing highs and lows

    Args:
        candles: List of (start_time, high, low) tuples
        window: Number of candles to check on each side

    Returns:
        (swing_highs, swing_lows) - Lists of (index, price) tuples
    """
    swing_highs = []
    swing_lows = []

    if len(candles) < window * 2 + 1:
        return swing_highs, swing_lows

    for i in range(window, len(candles) - window):
        current_high = candles[i][1]  # high price
        current_low = candles[i][2]   # low price

        # Check if it's a swing high
        is_swing_high = True
        for j in range(i - window, i + window + 1):
            if j != i and candles[j][1] >= current_high:
                is_swing_high = False
                break

        if is_swing_high:
            swing_highs.append((i, current_high, candles[i][0]))

        # Check if it's a swing low
        is_swing_low = True
        for j in range(i - window, i + window + 1):
            if j != i and candles[j][2] <= current_low:
                is_swing_low = False
                break

        if is_swing_low:
            swing_lows.append((i, current_low, candles[i][0]))

    return swing_highs, swing_lows

def find_nearest_swing_levels(entry_price: float, swing_highs: List, swing_lows: List) -> Tuple[Optional[float], Optional[float]]:
    """
    Find nearest swing high above and swing low below entry price

    Returns:
        (nearest_swing_high, nearest_swing_low)
    """
    # Find nearest swing high ABOVE entry
    highs_above = [price for _, price, _ in swing_highs if price > entry_price]
    nearest_high = min(highs_above) if highs_above else None

    # Find nearest swing low BELOW entry
    lows_below = [price for _, price, _ in swing_lows if price < entry_price]
    nearest_low = max(lows_below) if lows_below else None

    return nearest_high, nearest_low

def analyze_trade(conn, trade_data: Tuple) -> Dict:
    """Analyze a single trade for swing level proximity"""
    trade_id, epic, direction, entry_price, sl_price, tp_price, timestamp, pnl, outcome = trade_data

    # Parse timestamp
    entry_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')

    # Query candles before entry (100 candles on 5m = 8.3 hours)
    lookback_hours = 12  # Extra buffer for weekends
    start_time = entry_time - timedelta(hours=lookback_hours)

    cursor = conn.cursor()

    # Get 5-minute candles before entry
    query = """
        SELECT start_time, high, low, close
        FROM ig_candles
        WHERE epic = %s
        AND timeframe = 5
        AND start_time < %s
        AND start_time >= %s
        ORDER BY start_time ASC
    """

    cursor.execute(query, (epic, entry_time, start_time))
    candles = cursor.fetchall()
    cursor.close()

    result = {
        'trade_id': trade_id,
        'epic': epic,
        'direction': direction,
        'entry_price': entry_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'timestamp': timestamp,
        'pnl': pnl,
        'outcome': outcome,
        'candles_found': len(candles),
        'nearest_swing_high': None,
        'nearest_swing_low': None,
        'distance_to_high_pips': None,
        'distance_to_low_pips': None,
        'near_swing_high': False,
        'near_swing_low': False,
        'swing_context': 'INSUFFICIENT_DATA'
    }

    if len(candles) < 20:
        return result

    # Detect swings
    swing_highs, swing_lows = detect_swings(candles, window=3)

    if not swing_highs and not swing_lows:
        result['swing_context'] = 'NO_SWINGS_DETECTED'
        return result

    # Find nearest swing levels
    nearest_high, nearest_low = find_nearest_swing_levels(entry_price, swing_highs, swing_lows)

    result['nearest_swing_high'] = nearest_high
    result['nearest_swing_low'] = nearest_low

    # Calculate distances in pips
    if nearest_high:
        distance_high = calculate_pips(nearest_high - entry_price, epic)
        result['distance_to_high_pips'] = round(distance_high, 1)
        result['near_swing_high'] = distance_high <= 15

    if nearest_low:
        distance_low = calculate_pips(entry_price - nearest_low, epic)
        result['distance_to_low_pips'] = round(distance_low, 1)
        result['near_swing_low'] = distance_low <= 15

    # Determine swing context
    if direction == 'BUY':
        if result['near_swing_high']:
            result['swing_context'] = 'BUY_AT_RESISTANCE'
        elif result['near_swing_low']:
            result['swing_context'] = 'BUY_AT_SUPPORT'
        else:
            result['swing_context'] = 'BUY_AWAY_FROM_SWINGS'
    else:  # SELL
        if result['near_swing_low']:
            result['swing_context'] = 'SELL_AT_SUPPORT'
        elif result['near_swing_high']:
            result['swing_context'] = 'SELL_AT_RESISTANCE'
        else:
            result['swing_context'] = 'SELL_AWAY_FROM_SWINGS'

    return result

def main():
    """Main analysis function"""
    print("=" * 120)
    print("SWING LEVEL ANALYSIS - LATEST 20 CLOSED TRADES")
    print("=" * 120)
    print()

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to database\n")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        sys.exit(1)

    # Analyze each trade
    results = []
    for i, trade in enumerate(trades_data, 1):
        print(f"Analyzing trade {i}/20 (ID: {trade[0]})...", end=' ')
        result = analyze_trade(conn, trade)
        results.append(result)
        print(f"✓ {result['candles_found']} candles, context: {result['swing_context']}")

    conn.close()

    # Print detailed results table
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)
    print()

    header = f"{'ID':<6} {'Epic':<22} {'Dir':<5} {'Entry':<10} {'High':<10} {'→High':<8} {'Low':<10} {'→Low':<8} {'Outcome':<8} {'P&L':<10} {'Context':<25}"
    print(header)
    print("-" * 120)

    for r in results:
        high_str = f"{r['nearest_swing_high']:.5f}" if r['nearest_swing_high'] else "N/A"
        low_str = f"{r['nearest_swing_low']:.5f}" if r['nearest_swing_low'] else "N/A"
        dist_high = f"{r['distance_to_high_pips']:.1f}" if r['distance_to_high_pips'] else "N/A"
        dist_low = f"{r['distance_to_low_pips']:.1f}" if r['distance_to_low_pips'] else "N/A"

        # Shorten epic name
        epic_short = r['epic'].replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

        row = f"{r['trade_id']:<6} {epic_short:<22} {r['direction']:<5} {r['entry_price']:<10.5f} {high_str:<10} {dist_high:<8} {low_str:<10} {dist_low:<8} {r['outcome']:<8} {r['pnl']:<10.2f} {r['swing_context']:<25}"
        print(row)

    # Statistical analysis
    print("\n" + "=" * 120)
    print("PATTERN ANALYSIS")
    print("=" * 120)
    print()

    # Filter valid results (with swing data)
    valid_results = [r for r in results if r['swing_context'] not in ['INSUFFICIENT_DATA', 'NO_SWINGS_DETECTED']]

    if not valid_results:
        print("✗ Insufficient data for statistical analysis")
        return

    # Count patterns
    buy_at_resistance = [r for r in valid_results if r['swing_context'] == 'BUY_AT_RESISTANCE']
    buy_at_support = [r for r in valid_results if r['swing_context'] == 'BUY_AT_SUPPORT']
    buy_away = [r for r in valid_results if r['swing_context'] == 'BUY_AWAY_FROM_SWINGS']
    sell_at_support = [r for r in valid_results if r['swing_context'] == 'SELL_AT_SUPPORT']
    sell_at_resistance = [r for r in valid_results if r['swing_context'] == 'SELL_AT_RESISTANCE']
    sell_away = [r for r in valid_results if r['swing_context'] == 'SELL_AWAY_FROM_SWINGS']

    def calc_win_rate(trades):
        if not trades:
            return 0, 0, 0.0
        wins = len([t for t in trades if t['outcome'] == 'WIN'])
        total = len(trades)
        return wins, total, (wins / total * 100) if total > 0 else 0.0

    print("1. BUY TRADES AT RESISTANCE (Swing High)")
    wins, total, rate = calc_win_rate(buy_at_resistance)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    if buy_at_resistance:
        avg_dist = sum([r['distance_to_high_pips'] for r in buy_at_resistance if r['distance_to_high_pips']]) / len(buy_at_resistance)
        print(f"   Avg distance to resistance: {avg_dist:.1f} pips")
    print()

    print("2. BUY TRADES AT SUPPORT (Swing Low)")
    wins, total, rate = calc_win_rate(buy_at_support)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    if buy_at_support:
        avg_dist = sum([r['distance_to_low_pips'] for r in buy_at_support if r['distance_to_low_pips']]) / len(buy_at_support)
        print(f"   Avg distance to support: {avg_dist:.1f} pips")
    print()

    print("3. BUY TRADES AWAY FROM SWINGS")
    wins, total, rate = calc_win_rate(buy_away)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    print()

    print("4. SELL TRADES AT SUPPORT (Swing Low)")
    wins, total, rate = calc_win_rate(sell_at_support)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    if sell_at_support:
        avg_dist = sum([r['distance_to_low_pips'] for r in sell_at_support if r['distance_to_low_pips']]) / len(sell_at_support)
        print(f"   Avg distance to support: {avg_dist:.1f} pips")
    print()

    print("5. SELL TRADES AT RESISTANCE (Swing High)")
    wins, total, rate = calc_win_rate(sell_at_resistance)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    if sell_at_resistance:
        avg_dist = sum([r['distance_to_high_pips'] for r in sell_at_resistance if r['distance_to_high_pips']]) / len(sell_at_resistance)
        print(f"   Avg distance to resistance: {avg_dist:.1f} pips")
    print()

    print("6. SELL TRADES AWAY FROM SWINGS")
    wins, total, rate = calc_win_rate(sell_away)
    print(f"   Count: {total}, Wins: {wins}, Win Rate: {rate:.1f}%")
    print()

    # Overall statistics
    print("=" * 120)
    print("OVERALL STATISTICS")
    print("=" * 120)
    print()

    total_wins = len([r for r in valid_results if r['outcome'] == 'WIN'])
    total_trades = len(valid_results)
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

    print(f"Total analyzed trades: {total_trades}")
    print(f"Total wins: {total_wins}")
    print(f"Total losses: {total_trades - total_wins}")
    print(f"Overall win rate: {overall_win_rate:.1f}%")
    print()

    # Bad entry patterns (counter-trend)
    bad_entries = len(buy_at_resistance) + len(sell_at_support)
    good_entries = len(buy_at_support) + len(sell_at_resistance)
    neutral_entries = len(buy_away) + len(sell_away)

    print(f"Bad entries (BUY at resistance / SELL at support): {bad_entries}")
    print(f"Good entries (BUY at support / SELL at resistance): {good_entries}")
    print(f"Neutral entries (away from swings): {neutral_entries}")
    print()

    # Key findings
    print("=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)
    print()

    if len(buy_at_resistance) > 0:
        print(f"⚠ WARNING: {len(buy_at_resistance)} BUY trades taken within 15 pips of swing HIGH (resistance)")
        _, _, bar_wr = calc_win_rate(buy_at_resistance)
        print(f"  → Win rate: {bar_wr:.1f}% (likely poor risk/reward)")
        print()

    if len(sell_at_support) > 0:
        print(f"⚠ WARNING: {len(sell_at_support)} SELL trades taken within 15 pips of swing LOW (support)")
        _, _, sar_wr = calc_win_rate(sell_at_support)
        print(f"  → Win rate: {sar_wr:.1f}% (likely poor risk/reward)")
        print()

    if len(buy_at_support) > 0:
        print(f"✓ GOOD: {len(buy_at_support)} BUY trades taken near support")
        _, _, bas_wr = calc_win_rate(buy_at_support)
        print(f"  → Win rate: {bas_wr:.1f}%")
        print()

    if len(sell_at_resistance) > 0:
        print(f"✓ GOOD: {len(sell_at_resistance)} SELL trades taken near resistance")
        _, _, sar_wr = calc_win_rate(sell_at_resistance)
        print(f"  → Win rate: {sar_wr:.1f}%")
        print()

if __name__ == '__main__':
    main()
