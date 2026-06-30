#!/usr/bin/env python3
"""
EXPERIMENTAL (Jun 30 2026) — Faithful functional proxy of the LuxAlgo workflow,
to test whether the "it works on forex" impression survives honest, costed,
non-repainting simulation. Source is locked; this rebuilds BEHAVIOR, like the
June SuperTrend reverse-engineer.

Stack (how LuxAlgo strategies actually combine the two toolkits):
  - Smart Trail (Signals & Overlays)  -> ATR SuperTrend-style trend line + direction
  - Money Flow (Oscillator Matrix)    -> MFI on tick-volume (ltv) — the genuinely
                                          less-tested ingredient (everything prior was price-only)
  - Divergence (Oscillator Matrix)    -> regular bull/bear divergence price vs Money Flow,
                                          CONFIRMED at pivot+w (non-repainting) = the trigger
  - Confluence signal = divergence trigger ALIGNED with Smart Trail trend

Exit tested two ways: (A) Smart Trail flip (faithful trailing), (B) fixed ATR bracket.
Edge test: forward NET return (after spread) vs a random-entry null + trade-sim PF, IS/OOS.

Tick-volume (ltv) only exists ~Oct2025+, so window ~9 months. Run inside task-worker:
  python /app/forex_scanner/luxalgo_proxy.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import pip, cost, PIP_SIZE, _dbm, _wilder  # noqa: E402

PAIRS = ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
         'CS.D.AUDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.AUDUSD.MINI.IP',
         'CS.D.USDCAD.MINI.IP', 'CS.D.NZDUSD.MINI.IP']


def load(epic, tf=60):
    df = pd.read_sql(
        """SELECT start_time, open, high, low, close, ltv
           FROM ig_candles_backtest
           WHERE epic=%(e)s AND timeframe=%(tf)s AND ltv > 0
           ORDER BY start_time""",
        _dbm().get_engine(), params={'e': epic, 'tf': tf})
    if len(df) < 300:
        return None
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype(float)
    df['ltv'] = df['ltv'].astype(float)
    df['start_time'] = pd.to_datetime(df['start_time'])
    return df.reset_index(drop=True)


def atr(df, n=14):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return _wilder(tr, n)


def supertrend(df, period=10, mult=3.0):
    """Smart Trail proxy: ATR SuperTrend line + trend direction (+1/-1)."""
    a = atr(df, period)
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    hl2 = (h + l) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    n = len(df)
    fu = np.full(n, np.nan); fl = np.full(n, np.nan); dirn = np.ones(n, int)
    fu[0], fl[0] = upper[0], lower[0]
    for i in range(1, n):
        fu[i] = upper[i] if (upper[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lower[i] if (lower[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
        if dirn[i-1] == 1:
            dirn[i] = -1 if c[i] <= fl[i] else 1
        else:
            dirn[i] = 1 if c[i] >= fu[i] else -1
    return dirn


def mfi(df, n=14):
    """Money Flow Index using tick-volume (ltv)."""
    h, l, c, v = df['high'].values, df['low'].values, df['close'].values, df['ltv'].values
    tp = (h + l + c) / 3
    rmf = tp * v
    dtp = np.diff(tp, prepend=tp[0])
    pos = np.where(dtp > 0, rmf, 0.0)
    neg = np.where(dtp < 0, rmf, 0.0)
    out = np.full(len(df), np.nan)
    for i in range(n, len(df)):
        ps = pos[i-n+1:i+1].sum(); ns = neg[i-n+1:i+1].sum()
        out[i] = 100 - 100 / (1 + ps / ns) if ns > 0 else 100.0
    return out


def pivots(arr, w):
    """Return arrays of pivot-low and pivot-high indices (confirmed at i, value at i-w)."""
    lows, highs = [], []
    for i in range(w, len(arr) - w):
        seg = arr[i-w:i+w+1]
        if arr[i] == seg.min():
            lows.append(i)
        if arr[i] == seg.max():
            highs.append(i)
    return lows, highs


def rsi_osc(df, n=14):
    c = df['close'].values
    d = np.diff(c, prepend=c[0])
    g = _wilder(np.where(d > 0, d, 0.0), n)
    l = _wilder(np.where(d < 0, -d, 0.0), n)
    rs = g / np.where(l == 0, np.nan, l)
    return 100 - 100 / (1 + rs)


def signals(df, w=5, st_period=10, st_mult=3.0, osc_kind='mfi'):
    """LuxAlgo-style confluence signals. Returns list of (confirm_idx, dir)."""
    low = df['low'].values; high = df['high'].values
    osc = mfi(df) if osc_kind == 'mfi' else rsi_osc(df)
    dirn = supertrend(df, st_period, st_mult)
    plo, phi = pivots(low, w)
    _, osc_phi = pivots(high, w)  # unused; structure parity
    sigs = []
    # bullish regular divergence: price lower-low, MFI higher-low -> confirmed at pivot+w
    for a, b in zip(plo, plo[1:]):
        if np.isnan(osc[a]) or np.isnan(osc[b]):
            continue
        if low[b] < low[a] and osc[b] > osc[a]:
            ci = b + w  # confirmation bar (non-repainting)
            if ci < len(df) and dirn[ci] == 1:   # trend confluence
                sigs.append((ci, 'BUY'))
    plo2, phi2 = pivots(high, w)
    for a, b in zip(phi2, phi2[1:]):
        if np.isnan(osc[a]) or np.isnan(osc[b]):
            continue
        if high[b] > high[a] and osc[b] < osc[a]:
            ci = b + w
            if ci < len(df) and dirn[ci] == -1:
                sigs.append((ci, 'SELL'))
    return sorted(set(sigs)), dirn


def sim_trail(sigs, df, dirn, epic, H_cap=200):
    """Enter at confirm bar close, exit on Smart Trail flip against position (or cap)."""
    ps = pip(epic); cst = cost(epic)
    c = df['close'].values
    trades = []
    for ci, d in sigs:
        s = 1 if d == 'BUY' else -1
        entry = c[ci]
        exit_i = min(ci + H_cap, len(df) - 1)
        for j in range(ci + 1, min(ci + H_cap, len(df))):
            if (s == 1 and dirn[j] == -1) or (s == -1 and dirn[j] == 1):
                exit_i = j; break
        pips = s * (c[exit_i] - entry) / ps - cst
        trades.append((df['start_time'].iloc[ci], pips))
    return trades


def fwd_net(sigs, df, epic, H=24):
    ps = pip(epic); cst = cost(epic); c = df['close'].values
    out = []
    for ci, d in sigs:
        if ci + H >= len(df):
            continue
        s = 1 if d == 'BUY' else -1
        out.append((df['start_time'].iloc[ci], s * (c[ci+H] - c[ci]) / ps - cst))
    return out


def null_fwd(df, epic, n, H=24, seed=11):
    rng = np.random.default_rng(seed)
    ps = pip(epic); cst = cost(epic); c = df['close'].values
    out = []
    for _ in range(n):
        i = int(rng.integers(50, len(df) - H - 1))
        s = 1 if rng.random() > 0.5 else -1
        out.append(s * (c[i+H] - c[i]) / ps - cst)
    return out


def summ(x):
    x = np.array([v for v in x if not np.isnan(v)], float)
    if len(x) == 0:
        return "n=0"
    wins = x[x > 0]; loss = x[x <= 0]
    pf = wins.sum() / abs(loss.sum()) if loss.sum() != 0 else float('inf')
    return f"n={len(x):4d} mean={x.mean():7.2f} %pos={100*(x>0).mean():4.1f} pf={pf:4.2f}"


def run_variant(osc_kind):
    all_fwd, all_null, all_trail, trail_ts = [], [], [], []
    per_pair = []
    for epic in PAIRS:
        df = load(epic)
        if df is None:
            continue
        sigs, dirn = signals(df, osc_kind=osc_kind)
        fw = fwd_net(sigs, df, epic, 24)
        nl = null_fwd(df, epic, 2000, 24)
        tr = sim_trail(sigs, df, dirn, epic)
        all_fwd += [p for _,p in fw]; all_null += nl; all_trail += [p for _,p in tr]
        trail_ts += tr
        per_pair.append((epic, len(sigs), summ([p for _,p in fw]), summ([p for _,p in tr])))
    return all_fwd, all_null, all_trail, trail_ts, per_pair


def main():
    print(f"\n{'='*94}\nLUXALGO PROXY — SmartTrail + divergence confluence, ABLATION: "
          f"MoneyFlow(tick-vol MFI) vs price-only RSI\n  non-repainting, real costs, tick-vol window "
          f"(~Oct2025-Jun2026)\n{'='*94}")
    for osc_kind in ('mfi', 'rsi'):
        fwd, nul, trail, trail_ts, per_pair = run_variant(osc_kind)
        label = 'MONEY FLOW (tick-volume)' if osc_kind == 'mfi' else 'RSI (price-only ablation)'
        print(f"\n######## OSCILLATOR = {label} ########")
        for epic, ns, fs, ts in per_pair:
            print(f"   {epic:24s} sig={ns:3d}  fwd24[{fs}]  trail[{ts}]")
        print(f"   {'POOLED fwd24':24s}      [{summ(fwd)}]   NULL[{summ(nul)}]")
        if trail_ts:
            trail_ts.sort()
            split = trail_ts[int(len(trail_ts)*0.6)][0]
            isd = [p for t,p in trail_ts if t < split]; oos = [p for t,p in trail_ts if t >= split]
            print(f"   {'POOLED trail':24s}      [{summ(trail)}]")
            print(f"   {'  trail IS/OOS':24s}    IS[{summ(isd)}]  OOS[{summ(oos)}]")


if __name__ == '__main__':
    main()
