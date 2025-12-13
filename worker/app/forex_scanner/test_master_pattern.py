#!/usr/bin/env python3
"""
Master Pattern Strategy Backtest Script
Tests the ICT Power of 3 (AMD) strategy on historical data.
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, '/app')
os.chdir('/app')

import logging
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import psycopg2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host='postgres',
        port=5432,
        database='forex',
        user='postgres',
        password='postgres'
    )


def get_candles_from_db(epic: str, resolution: str = '5m', limit: int = 5000) -> Optional[pd.DataFrame]:
    """Fetch candles directly from database."""
    try:
        conn = get_db_connection()

        # Map resolution to interval
        interval_map = {
            '5m': '5 minute',
            '15m': '15 minute',
            '1h': '1 hour',
            '4h': '4 hour',
        }
        interval = interval_map.get(resolution, '5 minute')

        if resolution == '5m':
            # Use raw 5m data from ig_candles
            query = """
                SELECT start_time as timestamp, open, high, low, close, volume
                FROM ig_candles
                WHERE epic = %s
                ORDER BY start_time DESC
                LIMIT %s
            """
            df = pd.read_sql(query, conn, params=(epic, limit))
        else:
            # Resample from 5m data
            query = """
                SELECT start_time as timestamp, open, high, low, close, volume
                FROM ig_candles
                WHERE epic = %s
                ORDER BY start_time DESC
                LIMIT %s
            """
            df = pd.read_sql(query, conn, params=(epic, limit * 3))

            if len(df) == 0:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Resample
            resample_map = {'15m': '15min', '1h': '1h', '4h': '4h'}
            resample_freq = resample_map.get(resolution, '15min')

            df = df.resample(resample_freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            conn.close()
            return df.iloc[-limit:] if len(df) > limit else df

        conn.close()

        if len(df) == 0:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        return df

    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return None


def run_master_pattern_backtest(epic: str, pair: str, days: int = 30):
    """Run Master Pattern backtest on a single pair."""

    print(f"\n{'='*70}")
    print(f"MASTER PATTERN (ICT Power of 3) BACKTEST")
    print(f"Pair: {pair} ({epic})")
    print(f"Period: {days} days")
    print(f"{'='*70}\n")

    # Import strategy here to avoid import issues
    from forex_scanner.core.strategies.master_pattern_strategy import MasterPatternStrategy

    # Initialize strategy
    strategy = MasterPatternStrategy()

    # Fetch data
    logger.info("Fetching historical data...")
    df_5m = get_candles_from_db(epic, resolution='5m', limit=days * 288)  # 288 5m candles per day
    df_15m = get_candles_from_db(epic, resolution='15m', limit=days * 96)  # 96 15m candles per day

    if df_5m is None or len(df_5m) < 500:
        logger.error(f"Not enough 5m data for {pair}")
        return []

    logger.info(f"5m candles: {len(df_5m)}")
    logger.info(f"15m candles: {len(df_15m) if df_15m is not None else 0}")
    logger.info(f"Date range: {df_5m.index[0]} to {df_5m.index[-1]}")
    print()

    # Run backtest - iterate through data
    signals = []
    current_date = None
    days_processed = 0
    candles_processed = 0

    # Start after enough history
    start_idx = min(500, len(df_5m) // 2)

    for i in range(start_idx, len(df_5m)):
        current_time = df_5m.index[i]
        candles_processed += 1

        # Check if new day
        ts_date = current_time.date() if hasattr(current_time, 'date') else pd.Timestamp(current_time).date()

        if ts_date != current_date:
            current_date = ts_date
            days_processed += 1
            # Reset strategy state for new day (as it would in live trading)
            strategy.phase_tracker.check_daily_reset(pair, current_date)
            if days_processed % 5 == 0:
                logger.info(f"Day {days_processed}: {current_date}")

        # Get slice of data up to current point
        df_5m_slice = df_5m.iloc[max(0, i-200):i+1]

        # Get corresponding 15m data
        df_15m_slice = None
        if df_15m is not None:
            cutoff_time = df_5m_slice.index[-1]
            df_15m_mask = df_15m.index <= cutoff_time
            df_15m_slice = df_15m[df_15m_mask].copy()
            if len(df_15m_slice) > 100:
                df_15m_slice = df_15m_slice.iloc[-100:]

        # Detect signal - strategy uses df parameter, not df_5m
        try:
            signal = strategy.detect_signal(
                df=df_5m_slice,
                epic=epic,
                spread_pips=1.5,
                timeframe='5m'
            )

            if signal:
                # Handle both dict and object returns
                if isinstance(signal, dict):
                    sig_direction = signal.get('signal_type', signal.get('direction', 'UNKNOWN'))
                    sig_entry = signal.get('price', signal.get('entry_price', 0))
                    sig_sl = signal.get('stop_loss', 0)
                    sig_tp = signal.get('take_profit', 0)
                    sig_conf = signal.get('confidence', 0)
                else:
                    sig_direction = signal.direction
                    sig_entry = signal.entry_price
                    sig_sl = signal.stop_loss
                    sig_tp = signal.take_profit
                    sig_conf = signal.confidence

                signals.append({
                    'timestamp': current_time,
                    'direction': sig_direction,
                    'entry': sig_entry,
                    'sl': sig_sl,
                    'tp': sig_tp,
                    'confidence': sig_conf,
                    'signal': signal
                })

                print()
                print(f"{'🟢' if sig_direction == 'BULL' else '🔴'} SIGNAL at {current_time}")
                print(f"   Direction: {sig_direction}")
                print(f"   Entry: {sig_entry:.5f}")
                print(f"   SL: {sig_sl:.5f}")
                print(f"   TP: {sig_tp:.5f}")
                print(f"   Confidence: {sig_conf:.1%}")
                print()
        except Exception as e:
            logger.error(f"Error at {current_time}: {e}")
            continue

    # Print summary
    print()
    print(f"{'='*70}")
    print(f"BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"Days processed: {days_processed}")
    print(f"Candles processed: {candles_processed}")
    print(f"Total signals: {len(signals)}")

    if signals:
        bull_signals = sum(1 for s in signals if s['direction'] == 'BULL')
        bear_signals = sum(1 for s in signals if s['direction'] == 'BEAR')
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)

        print(f"  - Bull signals: {bull_signals}")
        print(f"  - Bear signals: {bear_signals}")
        print(f"  - Avg confidence: {avg_confidence:.1%}")

        print()
        print("SIGNALS:")
        print("-" * 70)
        for s in signals:
            risk_pips = abs(s['entry'] - s['sl']) / 0.0001
            reward_pips = abs(s['tp'] - s['entry']) / 0.0001
            rr = reward_pips / risk_pips if risk_pips > 0 else 0
            print(f"{s['timestamp']} | {s['direction']:4} | Entry: {s['entry']:.5f} | "
                  f"SL: {s['sl']:.5f} | TP: {s['tp']:.5f} | R:R: {rr:.1f} | Conf: {s['confidence']:.0%}")

    return signals


def main():
    """Main entry point."""
    # Default settings
    pair = 'EURUSD'
    epic = 'CS.D.EURUSD.CEEM.IP'
    days = 30

    # Parse command line args
    if len(sys.argv) > 1:
        pair = sys.argv[1].upper()
        epic_map = {
            'EURUSD': 'CS.D.EURUSD.CEEM.IP',
            'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
            'USDJPY': 'CS.D.USDJPY.MINI.IP',
            'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
            'USDCAD': 'CS.D.USDCAD.MINI.IP',
            'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
            'EURJPY': 'CS.D.EURJPY.MINI.IP',
            'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
        }
        epic = epic_map.get(pair, f'CS.D.{pair}.MINI.IP')

    if len(sys.argv) > 2:
        try:
            days = int(sys.argv[2])
        except ValueError:
            pass

    # Run backtest
    signals = run_master_pattern_backtest(epic, pair, days)

    return 0 if signals else 1


if __name__ == '__main__':
    sys.exit(main())
