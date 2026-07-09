#!/usr/bin/env python3
"""
EXPERIMENTAL (Jul 6 2026) — Replicate LuxAlgo AI-backtester strategy #2 (USDJPY):

  Long : Confirmation STRONG Bullish + Smart Trail BEARISH + Neo Cloud Bullish
  Short: mirror. Exit: Confirmation built-in exits. Unit size, frictionless.
  Claimed since Mar 11 2026: 47 closed trades, WR 97.87%, PF 29.79, avg +0.113 JPY
  (~11.3 pips). Open long from Jun 11 @160.532 (excluded from their stats).

Proxies (source locked, behavior rebuild):
  Confirmation STRONG : EMA9/21 cross + RSI>55 (bull) / <45 (bear)   [EMAXs]
                        smoothed-HA flip + body>0.5*ATR              [sHAs]
  Smart Trail         : SuperTrend(10,3) direction
  Neo Cloud           : EMA21 vs EMA55 cloud sign
  Built-in exits      : E-opp  = opposite plain confirmation (EMA9/21 cross back)
                        E-trail= chandelier ATR(10)x2 trailing stop
                        E-prof = MECHANISM NULL: exit at first close in profit,
                                 no stop (this is what manufactures 97% WR)

Tests:
  A) eval window (>= Mar 11 2026), zero cost: trade count / WR / PF / avg per combo;
     unrealized open-trade P&L reported separately (their stats hide it).
  B) mechanism: RANDOM entries + each exit style, zero cost — how much WR does the
     exit alone buy?
  C) truth: full 2020-2026, real costs, IS/OOS, entry vs random-with-same-exit.

Run: docker exec task-worker python /app/forex_scanner/luxalgo_neocloud_rep.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import pip, cost, _dbm  # noqa: E402
from luxalgo_proxy import atr, supertrend, rsi_osc  # noqa: E402
from luxalgo_hyperwave_rep import load, ema  # noqa: E402

EVAL_START = pd.Timestamp('2026-03-11')
EPIC = 'CS.D.USDJPY.MINI.IP'


# ---------- strong confirmation proxies: [(idx, +1|-1)] ----------

def strong_emax(df):
    c = df['close'].values
    f, s = ema(c, 9), ema(c, 21)
    r = rsi_osc(df, 14)
    out = []
    for i in range(1, len(c)):
        if f[i] > s[i] and f[i-1] <= s[i-1] and r[i] > 55:
            out.append((i, +1))
        elif f[i] < s[i] and f[i-1] >= s[i-1] and r[i] < 45:
            out.append((i, -1))
    return out


def strong_sha(df):
    o = ema(df['open'].values, 6); h = ema(df['high'].values, 6)
    l = ema(df['low'].values, 6); c = ema(df['close'].values, 6)
    n = len(c); a = atr(df, 14)
    ha_c = (o + h + l + c) / 4
    ha_o = np.empty(n); ha_o[0] = o[0]
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
    col = np.where(ha_c > ha_o, 1, -1)
    out = []
    for i in range(1, n):
        if np.isnan(a[i]):
            continue
        big = abs(ha_c[i] - ha_o[i]) > 0.5 * a[i]
        if col[i] == 1 and col[i-1] == -1 and big:
            out.append((i, +1))
        elif col[i] == -1 and col[i-1] == 1 and big:
            out.append((i, -1))
    return out


STRONG = {'EMAXs': strong_emax, 'sHAs': strong_sha}


def entries(df, strong_fn, need_trail_against=True):
    st = supertrend(df, 10, 3.0)
    cf, cs = ema(df['close'].values, 21), ema(df['close'].values, 55)
    cloud = np.where(cf > cs, 1, -1)
    out = []
    for i, s in strong_fn(df):
        if cloud[i] != s:
            continue
        if need_trail_against and st[i] != -s:   # Smart Trail AGAINST the entry dir
            continue
        out.append((i, s))
    return out


# ---------- exits ----------

def sim(ents, df, epic, cst, exit_kind, H_cap=5000):
    """One-position-at-a-time. Returns (closed trades [(t, pips, bars)], open_info)."""
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    t = df['start_time'].values
    a = atr(df, 10)
    f9, f21 = ema(c, 9), ema(c, 21)
    ps = pip(epic)
    trades = []
    i_free = 0
    open_info = None
    for ei, s in ents:
        if ei < i_free or ei >= len(c) - 1:
            continue
        entry = c[ei]
        exit_i = None
        if exit_kind == 'E-trail':
            peak = c[ei]
            for j in range(ei + 1, min(ei + H_cap, len(c))):
                peak = max(peak, h[j]) if s == 1 else min(peak, l[j])
                stop = peak - 2 * a[j] if s == 1 else peak + 2 * a[j]
                if (s == 1 and c[j] <= stop) or (s == -1 and c[j] >= stop):
                    exit_i = j; break
        elif exit_kind == 'E-opp':
            for j in range(ei + 1, min(ei + H_cap, len(c))):
                if s == 1 and f9[j] < f21[j] and f9[j-1] >= f21[j-1]:
                    exit_i = j; break
                if s == -1 and f9[j] > f21[j] and f9[j-1] <= f21[j-1]:
                    exit_i = j; break
        elif exit_kind == 'E-prof':
            for j in range(ei + 1, min(ei + H_cap, len(c))):
                if s * (c[j] - entry) / ps > cst + 1:   # first net-profitable close
                    exit_i = j; break
        if exit_i is None:
            open_info = (t[ei], s, entry, s * (c[-1] - entry) / ps - cst)
            i_free = len(c)
            continue
        trades.append((t[ei], s * (c[exit_i] - entry) / ps - cst, exit_i - ei))
        i_free = exit_i
    return trades, open_info


def summ(tr):
    x = np.array([p for _, p, _ in tr], float)
    if len(x) == 0:
        return "n=0"
    bars = np.array([b for _, _, b in tr], float)
    w = x[x > 0]; l = x[x <= 0]
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float('inf')
    return (f"n={len(x):4d} WR={100*(x>0).mean():5.1f}% PF={pf:6.2f} avg={x.mean():7.2f}p "
            f"tot={x.sum():8.0f}p medBars={np.median(bars):5.0f}")


def main():
    for tf in (15, 60):
        df = load(EPIC, tf)
        m = (df['start_time'] >= EVAL_START).values
        n80 = int(len(df) * 0.8)
        cst = cost(EPIC)
        print(f"\n{'='*100}\nTF={tf}m  {EPIC}  (target: 47 closed, WR 97.9%, PF 29.8, "
              f"avg +11.3p, frictionless)\n{'='*100}")
        for sname, sfn in STRONG.items():
            ents = entries(df, sfn)
            ents_w = [(i, s) for i, s in ents if m[i]]
            print(f"\n  entry={sname} (+trail-against +cloud-aligned): "
                  f"{len(ents)} signals all-history, {len(ents_w)} in eval window")
            for ek in ('E-opp', 'E-trail', 'E-prof'):
                tr, op = sim(ents_w, df, EPIC, 0.0, ek)
                ops = (f"  OPEN {'L' if op[1]==1 else 'S'}@{op[2]:.3f} "
                       f"unreal={op[3]:+.0f}p" if op else "")
                print(f"    [eval, zero cost] {ek:7s} {summ(tr)}{ops}")
            # truth: full history, real costs, IS/OOS, per exit
            for ek in ('E-opp', 'E-trail'):
                tr, _ = sim(ents, df, EPIC, cst, ek)
                isd = [x for x in tr if x[0] < df['start_time'].values[n80]]
                oos = [x for x in tr if x[0] >= df['start_time'].values[n80]]
                print(f"    [FULL, real cost] {ek:7s} ALL {summ(tr)} | OOS {summ(oos)}")
        # mechanism null: random entries, same exits, eval window, zero cost
        rng = np.random.default_rng(3)
        lo = int(np.argmax(m))
        n_r = 60
        idx = np.sort(rng.choice(np.arange(lo, len(df) - 10), size=n_r, replace=False))
        r_ents = [(int(i), int(rng.choice([-1, 1]))) for i in idx]
        print(f"\n  RANDOM entries (n={n_r}, eval window, zero cost) — exit mechanism alone:")
        for ek in ('E-opp', 'E-trail', 'E-prof'):
            tr, op = sim(r_ents, df, EPIC, 0.0, ek)
            ops = f"  OPEN unreal={op[3]:+.0f}p" if op else ""
            print(f"    {ek:7s} {summ(tr)}{ops}")


if __name__ == '__main__':
    main()
