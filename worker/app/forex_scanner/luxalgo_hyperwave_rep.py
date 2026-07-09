#!/usr/bin/env python3
"""
EXPERIMENTAL (Jul 6 2026) — Replicate the LuxAlgo AI-backtester strategy the user found:

  Long : trigger = Confirmation ANY BEARISH, filter = HyperWave < 50
  Short: trigger = Confirmation ANY BULLISH, filter = HyperWave > 50
  Exit : opposite qualified signal (stop-and-reverse, always-in once first trade opens)
  Size : unit. Claimed (JPY pair, eval Mar 10 2026+): 447 trades, WR 69.35%, PF 2.084,
         avg trade +0.051 JPY (~5.1 pips) — almost certainly FRICTIONLESS.

Source is locked, so like luxalgo_proxy.py we rebuild BEHAVIOR with multiple proxies
per component and demand robustness across them:

  Confirmation proxy A: SuperTrend(10,3) direction flip
  Confirmation proxy B: EMA9/EMA21 cross
  Confirmation proxy C: smoothed Heikin-Ashi candle color flip (closest to LuxAlgo's
                        published "confirmation" behavior per community reverse-engineering)
  HyperWave proxy    1: EMA-smoothed RSI(14) (0-100)
  HyperWave proxy    2: double-smoothed stochastic (0-100)

Evaluation:
  Phase 1 (match): eval window 2026-03-10 → data end, ZERO cost, per TF (5m/15m/60m),
                   per proxy combo — does anything reproduce ~5/day, WR~69%, PF~2?
  Phase 2 (truth): best-matching combo, full 2020-2026 history, IS/OOS split,
                   WITH real costs, vs random stop-and-reverse null at same frequency,
                   vs the INVERTED (trend-following) variant.

Run: docker exec task-worker python /app/forex_scanner/luxalgo_hyperwave_rep.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import pip, cost, _dbm, _wilder  # noqa: E402
from luxalgo_proxy import atr, supertrend, rsi_osc      # noqa: E402

EVAL_START = pd.Timestamp('2026-03-10')


def load(epic, tf):
    df = pd.read_sql(
        """SELECT start_time, open, high, low, close FROM ig_candles_backtest
           WHERE epic=%(e)s AND timeframe=%(tf)s ORDER BY start_time""",
        _dbm().get_engine(), params={'e': epic, 'tf': tf})
    if len(df) < 500:
        return None
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype(float)
    df['start_time'] = pd.to_datetime(df['start_time'])
    return df.reset_index(drop=True)


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


# ---------- Confirmation proxies: return arrays of (idx, 'BULL'|'BEAR') flips ----------

def conf_supertrend(df):
    d = supertrend(df, 10, 3.0)
    out = []
    for i in range(1, len(d)):
        if d[i] == 1 and d[i-1] == -1:
            out.append((i, 'BULL'))
        elif d[i] == -1 and d[i-1] == 1:
            out.append((i, 'BEAR'))
    return out


def conf_ema_cross(df):
    c = df['close'].values
    f, s = ema(c, 9), ema(c, 21)
    out = []
    for i in range(1, len(c)):
        if f[i] > s[i] and f[i-1] <= s[i-1]:
            out.append((i, 'BULL'))
        elif f[i] < s[i] and f[i-1] >= s[i-1]:
            out.append((i, 'BEAR'))
    return out


def conf_smooth_ha(df):
    """Smoothed Heikin-Ashi color flip: EMA(6)-pre-smoothed OHLC -> HA candles."""
    o = ema(df['open'].values, 6); h = ema(df['high'].values, 6)
    l = ema(df['low'].values, 6); c = ema(df['close'].values, 6)
    n = len(c)
    ha_c = (o + h + l + c) / 4
    ha_o = np.empty(n); ha_o[0] = o[0]
    for i in range(1, n):
        ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
    col = np.where(ha_c > ha_o, 1, -1)
    out = []
    for i in range(1, n):
        if col[i] == 1 and col[i-1] == -1:
            out.append((i, 'BULL'))
        elif col[i] == -1 and col[i-1] == 1:
            out.append((i, 'BEAR'))
    return out


CONFS = {'ST': conf_supertrend, 'EMAX': conf_ema_cross, 'sHA': conf_smooth_ha}


# ---------- HyperWave proxies: 0-100 oscillator ----------

def hw_rsi(df):
    return ema(rsi_osc(df, 14), 5)


def hw_stoch(df):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = 14
    hh = pd.Series(h).rolling(n).max().values
    ll = pd.Series(l).rolling(n).min().values
    rng = np.where(hh - ll == 0, np.nan, hh - ll)
    k = 100 * (c - ll) / rng
    return ema(ema(k, 3), 3)


HWS = {'rsi': hw_rsi, 'stoch': hw_stoch}


# ---------- Strategy: contrarian stop-and-reverse ----------

def entries(df, conf_fn, hw_fn, invert=False):
    """Qualified entry list [(idx, +1|-1)]. Contrarian per LuxAlgo config unless invert."""
    hw = hw_fn(df)
    out = []
    for i, kind in conf_fn(df):
        if np.isnan(hw[i]):
            continue
        if not invert:
            if kind == 'BEAR' and hw[i] < 50:
                out.append((i, +1))   # long on bearish confirmation, HW below 50
            elif kind == 'BULL' and hw[i] > 50:
                out.append((i, -1))
        else:
            if kind == 'BULL' and hw[i] > 50:
                out.append((i, +1))
            elif kind == 'BEAR' and hw[i] < 50:
                out.append((i, -1))
    return out


def sim_sar(ents, df, epic, cst):
    """Stop-and-reverse sim. Entry/exit at bar close of signal bar. Returns trade list
    of (entry_time, pips_net)."""
    c = df['close'].values
    t = df['start_time'].values
    trades = []
    pos = 0; e_i = None
    for i, s in ents:
        if s == pos:
            continue
        if pos != 0:
            trades.append((t[e_i], pos * (c[i] - c[e_i]) / pip(epic) - cst))
        pos = s; e_i = i
    return trades


def summ(pips_list):
    x = np.array(pips_list, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0)
    w = x[x > 0]; l = x[x <= 0]
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float('inf')
    return dict(n=len(x), wr=100 * (x > 0).mean(), pf=pf, avg=x.mean(), tot=x.sum())


def fmt(s):
    if s['n'] == 0:
        return "n=0"
    return (f"n={s['n']:4d} WR={s['wr']:5.1f}% PF={s['pf']:5.2f} "
            f"avg={s['avg']:6.2f}p tot={s['tot']:8.0f}p")


def phase1(epic):
    print(f"\n{'='*100}\nPHASE 1 — match LuxAlgo eval window (>= {EVAL_START.date()}), "
          f"ZERO COST, {epic}\n  target: ~447 trades (~5/day), WR 69.4%, PF 2.08, avg +5.1 pips"
          f"\n{'='*100}")
    best = None
    for tf in (5, 15, 60):
        df = load(epic, tf)
        if df is None:
            continue
        m = df['start_time'] >= EVAL_START
        for cname, cfn in CONFS.items():
            for hname, hfn in HWS.items():
                ents = entries(df, cfn, hfn)
                ents_w = [(i, s) for i, s in ents if m.iloc[i]]
                tr = sim_sar(ents_w, df, epic, 0.0)
                s = summ([p for _, p in tr])
                print(f"  tf={tf:3d} conf={cname:4s} hw={hname:5s}  {fmt(s)}")
                score = -abs(s['n'] - 447) - abs(s.get('pf', 0) - 2.08) * 50 if s['n'] else -9e9
                if best is None or score > best[0]:
                    best = (score, tf, cname, hname)
    return best[1], best[2], best[3]


def phase2(epic, tf, cname, hname):
    print(f"\n{'='*100}\nPHASE 2 — truth test: {epic} tf={tf} conf={cname} hw={hname}, "
          f"2020-2026 full history\n{'='*100}")
    df = load(epic, tf)
    cst = cost(epic)
    n80 = int(len(df) * 0.8)
    for label, cs in (('ZERO cost', 0.0), (f'REAL cost ({cst}p RT)', cst)):
        ents = entries(df, CONFS[cname], HWS[hname])
        tr = sim_sar(ents, df, epic, cs)
        isd = [p for t, p in tr if t < df['start_time'].values[n80]]
        oos = [p for t, p in tr if t >= df['start_time'].values[n80]]
        ev = [p for t, p in tr if t >= EVAL_START.to_datetime64()]
        print(f"  [{label:22s}] ALL  {fmt(summ([p for _, p in tr]))}")
        print(f"  [{label:22s}] IS80 {fmt(summ(isd))}")
        print(f"  [{label:22s}] OOS20{fmt(summ(oos))}")
        print(f"  [{label:22s}] eval {fmt(summ(ev))}")
    # inverted (trend-following) variant, zero cost — is the contrarian sign real?
    ents_inv = entries(df, CONFS[cname], HWS[hname], invert=True)
    tri = sim_sar(ents_inv, df, epic, 0.0)
    print(f"  [INVERTED, zero cost   ] ALL  {fmt(summ([p for _, p in tri]))}")
    # random-flip null at same frequency, zero cost
    rng = np.random.default_rng(7)
    n_sig = len(entries(df, CONFS[cname], HWS[hname]))
    idx = np.sort(rng.choice(np.arange(100, len(df) - 1), size=n_sig, replace=False))
    ents_r = [(int(i), int(rng.choice([-1, 1]))) for i in idx]
    trr = sim_sar(ents_r, df, epic, 0.0)
    print(f"  [RANDOM null, zero cost] ALL  {fmt(summ([p for _, p in trr]))}")


def main():
    epic = 'CS.D.USDJPY.MINI.IP'
    tf, cname, hname = phase1(epic)
    print(f"\n>>> best match: tf={tf} conf={cname} hw={hname}")
    phase2(epic, tf, cname, hname)
    # robustness: same combo on other JPY crosses + EURUSD, real cost
    print(f"\n{'='*100}\nPHASE 3 — cross-pair robustness (same combo, REAL costs, full history)"
          f"\n{'='*100}")
    for ep in ('CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP', 'CS.D.EURUSD.CEEM.IP',
               'CS.D.GBPUSD.MINI.IP'):
        df = load(ep, tf)
        if df is None:
            print(f"  {ep}: no data"); continue
        tr = sim_sar(entries(df, CONFS[cname], HWS[hname]), df, ep, cost(ep))
        n80 = int(len(df) * 0.8)
        oos = [p for t, p in tr if t >= df['start_time'].values[n80]]
        print(f"  {ep:26s} ALL {fmt(summ([p for _, p in tr]))}   OOS20 {fmt(summ(oos))}")


if __name__ == '__main__':
    main()
