#!/usr/bin/env python3
"""
LuxAlgo money-flow-divergence FORWARD-TEST MONITOR (monitor-only, Jun 30 2026).

Logs confirmed Smart-Trail + Money-Flow(tick-vol MFI) + divergence signals on the
pairs where the proxy showed entry skill (JPY + commodity), to accumulate genuine
out-of-sample data. NO execution, NO touching the live scanner. Idempotent.

Validation context: project_luxalgo_proxy_jun30 (proxy was breakeven OOS on 9mo;
money-flow gave a marginal real entry-skill lift). This monitor is the honest
forward test, since tick-volume only accrues going forward.

Reads LIVE 1m candles from ig_candles (ltv populated), resamples to 1h.
Run inside task-worker:
  python /app/forex_scanner/luxalgo_monitor.py --once      # single pass
  python /app/forex_scanner/luxalgo_monitor.py --loop      # loop every 30 min
"""
from __future__ import annotations
import sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import pip, cost, _dbm                 # noqa: E402
from luxalgo_proxy import supertrend, mfi, signals            # noqa: E402

# Pairs where the proxy concentrated its entry skill (JPY + commodity).
PAIRS = ['CS.D.EURJPY.MINI.IP', 'CS.D.USDJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP',
         'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP']
LOOKBACK_DAYS = 70
FWD_H = 24          # forward-eval horizon in 1h bars


def load_live_1h(epic: str) -> pd.DataFrame | None:
    """Read live 1m candles from ig_candles, resample to 1h (sum tick-volume)."""
    q = f"""
        SELECT start_time, open, high, low, close, ltv
        FROM ig_candles
        WHERE epic = %(e)s AND timeframe = 1
          AND start_time >= now() - interval '{int(LOOKBACK_DAYS)} days'
        ORDER BY start_time
    """
    df = pd.read_sql(q, _dbm().get_engine(), params={'e': epic})
    if len(df) < 2000:
        return None
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype(float)
    df['ltv'] = df['ltv'].fillna(0).astype(float)
    df = df.set_index(pd.to_datetime(df['start_time']))
    res = df.resample('1h').agg(open=('open', 'first'), high=('high', 'max'),
                                low=('low', 'min'), close=('close', 'last'),
                                ltv=('ltv', 'sum')).dropna(subset=['close'])
    res = res.reset_index().rename(columns={'index': 'start_time'})
    if 'start_time' not in res.columns:
        res = res.rename(columns={res.columns[0]: 'start_time'})
    return res if len(res) > 300 else None


def process(epic: str, conn) -> tuple[int, int]:
    df = load_live_1h(epic)
    if df is None:
        return 0, 0
    sigs, dirn = signals(df, osc_kind='mfi')
    osc = mfi(df)
    c = df['close'].values
    ts = df['start_time']
    ps = pip(epic); cst = cost(epic)
    cur = conn.cursor()

    # 1) insert newly-confirmed signals (idempotent)
    inserted = 0
    for ci, d in sigs:
        cur.execute(
            """INSERT INTO luxalgo_monitor_signals
                 (epic, bar_time, direction, entry_price, mfi, smart_trail_dir, osc_kind)
               VALUES (%s,%s,%s,%s,%s,%s,'mfi')
               ON CONFLICT (epic, bar_time, direction) DO NOTHING""",
            (epic, ts.iloc[ci].to_pydatetime(), d, float(c[ci]),
             float(osc[ci]) if not np.isnan(osc[ci]) else None, int(dirn[ci])))
        inserted += cur.rowcount

    # 2) backfill forward outcomes for matured rows still NULL
    bar_idx = {t.to_pydatetime(): i for i, t in enumerate(ts)}
    cur.execute("""SELECT id, bar_time, direction FROM luxalgo_monitor_signals
                   WHERE epic=%s AND fwd_24h_pips IS NULL""", (epic,))
    updated = 0
    for sid, bt, d in cur.fetchall():
        i = bar_idx.get(bt)
        if i is None or i + FWD_H >= len(df):
            continue
        s = 1 if d == 'BUY' else -1
        fwd = s * (c[i + FWD_H] - c[i]) / ps - cst
        # trail exit: hold until Smart Trail flips against position
        exit_i = len(df) - 1
        for j in range(i + 1, len(df)):
            if (s == 1 and dirn[j] == -1) or (s == -1 and dirn[j] == 1):
                exit_i = j; break
        trail = s * (c[exit_i] - c[i]) / ps - cst
        cur.execute(
            """UPDATE luxalgo_monitor_signals
               SET fwd_24h_pips=%s, trail_exit_pips=%s, trail_exit_time=%s
               WHERE id=%s""",
            (float(fwd), float(trail), ts.iloc[exit_i].to_pydatetime(), sid))
        updated += 1
    conn.commit(); cur.close()
    return inserted, updated


def run_once():
    conn = _dbm().get_connection()
    tot_i = tot_u = 0
    for epic in PAIRS:
        try:
            ins, upd = process(epic, conn)
            tot_i += ins; tot_u += upd
            print(f"  {epic:24s} new_signals={ins}  outcomes_filled={upd}")
        except Exception as e:
            conn.rollback()
            print(f"  {epic:24s} ERROR {e}")
    conn.close()
    # running scorecard
    sc = pd.read_sql(
        """SELECT COUNT(*) n,
                  COUNT(fwd_24h_pips) matured,
                  ROUND(AVG(fwd_24h_pips)::numeric,2) fwd_mean,
                  ROUND(100.0*AVG((fwd_24h_pips>0)::int)::numeric,1) fwd_pct_pos,
                  ROUND(AVG(trail_exit_pips)::numeric,2) trail_mean
           FROM luxalgo_monitor_signals""", _dbm().get_engine())
    print(f"  TOTAL new={tot_i} filled={tot_u} | scorecard: {sc.to_dict('records')[0]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--interval', type=int, default=1800)
    a = ap.parse_args()
    if a.loop:
        print(f"🔁 LuxAlgo monitor loop every {a.interval}s, pairs={len(PAIRS)}")
        while True:
            print(f"--- pass @ {pd.Timestamp.utcnow()} ---")
            run_once()
            time.sleep(a.interval)
    else:
        run_once()
