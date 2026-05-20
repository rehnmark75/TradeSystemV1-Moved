#!/usr/bin/env python3
"""
Diagnostic: correlate RANGE_FADE EURJPY backtest outcomes with 1h ADX.

Runs a 90d backtest, then queries backtest_signals table + computes 1h ADX
at each signal timestamp from ig_candles_backtest.

Run inside task-worker:
  python /app/forex_scanner/scripts/diag_range_fade_adx.py
"""

import sys
import logging
from datetime import datetime, timezone

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

# Keep backtest noise quiet but allow final summary through
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

EPIC = 'CS.D.GBPUSD.MINI.IP'
DAYS = 90


def compute_adx(rows):
    """Compute ADX(14) from list of (start_time, open, high, low, close) rows."""
    import pandas as pd
    import numpy as np

    if len(rows) < 20:
        return None

    df = pd.DataFrame(rows, columns=['start_time', 'open', 'high', 'low', 'close'])
    df = df.sort_values('start_time').reset_index(drop=True)
    n = 14

    df['h-l']  = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low']  - df['close'].shift(1))
    df['tr']   = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    df['+dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0), 0
    )
    df['-dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0), 0
    )

    atr  = df['tr'].ewm(alpha=1/n, adjust=False).mean()
    pdm  = df['+dm'].ewm(alpha=1/n, adjust=False).mean()
    mdm  = df['-dm'].ewm(alpha=1/n, adjust=False).mean()
    pdi  = 100 * pdm / atr
    mdi  = 100 * mdm / atr
    dx   = 100 * abs(pdi - mdi) / (pdi + mdi)
    adx  = dx.ewm(alpha=1/n, adjust=False).mean()
    return float(adx.iloc[-1])


def main():
    import psycopg2
    import pandas as pd
    from commands.enhanced_backtest_commands import EnhancedBacktestCommands

    print(f"\n🔬 RANGE_FADE EURJPY 90d — 1h ADX vs outcome\n")

    # Skip re-running the backtest — use the most recent execution already in DB.
    # To re-run, set RERUN = True at the top of this script.
    RERUN = True
    if RERUN:
        eb = EnhancedBacktestCommands()
        eb.run_enhanced_backtest(
            epic=EPIC,
            days=DAYS,
            strategy='RANGE_FADE',
            timeframe='5m',
            show_signals=False,
            return_results=False,
            quiet=True,
        )

    # --- 2. Fetch signals from DB ---
    conn = psycopg2.connect(host='postgres', dbname='forex', user='postgres', password='postgres')

    def qone(sql, params=()):
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def qall(sql, params=()):
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # Get the most recent execution_id for RANGE_FADE
    row = qone("""
        SELECT execution_id FROM backtest_signals
        WHERE strategy_name = 'RANGE_FADE'
        ORDER BY created_at DESC LIMIT 1
    """)

    if not row:
        print("❌ No RANGE_FADE signals found in backtest_signals")
        conn.close()
        return

    exec_id = row[0]
    print(f"Using execution_id: {exec_id}")

    signals = qall("""
        SELECT signal_timestamp, signal_type, trade_result, pips_gained, confidence_score
        FROM backtest_signals
        WHERE execution_id = %s AND epic = %s
        ORDER BY signal_timestamp
    """, (exec_id, EPIC))

    print(f"Signals in DB: {len(signals)}")
    if not signals:
        conn.close()
        return

    # --- 3. For each signal, compute 1h ADX from ig_candles_backtest ---
    rows_out = []
    for sig_ts, sig_type, trade_result, pips, conf in signals:
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.replace(tzinfo=timezone.utc)

        candle_rows = qall("""
            SELECT start_time, open, high, low, close
            FROM ig_candles_backtest
            WHERE epic = %s AND timeframe = 60 AND start_time <= %s
            ORDER BY start_time DESC LIMIT 60
        """, (EPIC, sig_ts))

        adx_1h = compute_adx(candle_rows)

        is_winner = str(trade_result).lower() in ('win', 'take_profit', 'tp_hit')
        outcome = 'WIN' if is_winner else 'LOSS'

        rows_out.append({
            'timestamp': sig_ts.strftime('%Y-%m-%d %H:%M'),
            'direction': sig_type,
            'outcome': outcome,
            'trade_result': trade_result,
            'pips': round(pips, 1) if pips else None,
            'conf': round(conf, 2) if conf else None,
            'adx_1h': round(adx_1h, 1) if adx_1h is not None else None,
        })

    conn.close()

    df = pd.DataFrame(rows_out)
    df_v = df[df['adx_1h'].notna()].copy()

    # --- 4. Print full table ---
    print("\n--- Per-signal table ---")
    print(f"{'Timestamp':>18}  {'Dir':4}  {'Out':4}  {'Pips':>6}  {'ADX1h':>6}")
    for _, r in df.iterrows():
        print(f"  {r['timestamp']:>16}  {str(r['direction']):4}  {r['outcome']:4}  "
              f"{str(r['pips']):>6}  {str(r['adx_1h']):>6}")

    # --- 5. ADX bucket summary ---
    if len(df_v) > 0:
        df_v = df_v.copy()
        df_v['adx_bucket'] = pd.cut(
            df_v['adx_1h'],
            bins=[0, 20, 25, 30, 35, 40, 50, 999],
            labels=['<20', '20-25', '25-30', '30-35', '35-40', '40-50', '>50']
        )
        summary = df_v.groupby('adx_bucket', observed=True).apply(
            lambda g: pd.Series({
                'n': len(g),
                'wins': g['outcome'].eq('WIN').sum(),
                'wr%': round(g['outcome'].eq('WIN').mean() * 100, 1),
                'avg_pips': round(g['pips'].mean(), 1) if g['pips'].notna().any() else None,
            })
        ).reset_index()

        print("\n--- Win rate by 1h ADX bucket ---")
        print(f"  {'ADX':>6}  {'n':>4}  {'wins':>5}  {'WR%':>6}  {'avg_pips':>9}")
        for _, r in summary.iterrows():
            print(f"  {r['adx_bucket']:>6}  {int(r['n']):>4}  {int(r['wins']):>5}  "
                  f"{r['wr%']:>5.1f}%  {str(r['avg_pips']):>9}")

        win_adx  = df_v[df_v['outcome'] == 'WIN']['adx_1h']
        loss_adx = df_v[df_v['outcome'] == 'LOSS']['adx_1h']
        print(f"\n  Winner avg 1h ADX: {win_adx.mean():.1f}  (n={len(win_adx)})")
        print(f"  Loser  avg 1h ADX: {loss_adx.mean():.1f}  (n={len(loss_adx)})")

        # --- 6. By direction ---
        print("\n--- By direction ---")
        for d in ['BUY', 'SELL']:
            sub = df_v[df_v['direction'] == d]
            if len(sub) == 0:
                continue
            wa = sub[sub['outcome'] == 'WIN']['adx_1h'].mean()
            la = sub[sub['outcome'] == 'LOSS']['adx_1h'].mean()
            wr = sub['outcome'].eq('WIN').mean() * 100
            print(f"  {d}: n={len(sub)}, WR={wr:.1f}%, "
                  f"winner_adx={wa:.1f}, loser_adx={la:.1f}")


if __name__ == '__main__':
    main()
