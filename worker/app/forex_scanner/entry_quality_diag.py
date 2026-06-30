#!/usr/bin/env python3
"""
EXPERIMENTAL (Jun 30 2026) — Entry-quality diagnostic.

For every emitted signal (alert_history) across strategies, measure the QUALITY
of the ENTRY itself, independent of any exit logic:
  - forward net return (pips, after spread) at horizons H bars
  - MFE / MAE (max favorable / adverse excursion)
  - vs a RANDOM-entry null baseline (random bar + random direction)

Then slice net return by context (session, ADX regime, HTF-EMA200 alignment, RSI)
to answer: do our entries have directional skill anywhere, or are they coin flips?

Run inside task-worker:
  python /app/forex_scanner/entry_quality_diag.py --min-sig 25 --horizon 24
"""
from __future__ import annotations
import sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

from exit_redesign_sim import (  # noqa: E402
    load_alert_signals, candles_cached, discover_epics, pip, cost, PIP_SIZE, _dbm,
)

HORIZONS = [6, 12, 24, 48]   # bars (1h) -> 6h,12h,1d,2d


def _sign(d): return 1 if d == 'BUY' else -1


def signal_rows(strategy: str, min_sig: int):
    """Yield dicts of per-signal forward stats + context features."""
    rows = []
    for epic, _ in discover_epics(strategy, min_sig):
        if epic not in PIP_SIZE:
            continue
        sigs = load_alert_signals(epic, strategy)
        if len(sigs) < min_sig:
            continue
        df = candles_cached(epic, 60)
        if df is None:
            continue
        ps = pip(epic); cst = cost(epic)
        ts = df['start_time'].values.astype('datetime64[ns]')
        c = df['close'].values; h = df['high'].values; l = df['low'].values
        adx = df['adx'].values; ema200 = df['ema200'].values; rsi = df['rsi'].values
        n = len(df)
        for sig_ts, d in sigs:
            i = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(sig_ts))))
            if i >= n - max(HORIZONS) - 1:
                continue
            s = _sign(d)
            ep = c[i]
            row = {'strategy': strategy, 'epic': epic, 'dir': d,
                   'ts': pd.Timestamp(sig_ts),
                   'hour': pd.Timestamp(sig_ts).hour,
                   'adx': adx[i], 'rsi': rsi[i],
                   'htf_aligned': (c[i] > ema200[i]) == (d == 'BUY')}
            for H in HORIZONS:
                seg_h = h[i + 1:i + 1 + H]; seg_l = l[i + 1:i + 1 + H]
                ret = s * (c[i + H] - ep) / ps - cst
                if s > 0:
                    mfe = (seg_h.max() - ep) / ps; mae = (ep - seg_l.min()) / ps
                else:
                    mfe = (ep - seg_l.min()) / ps; mae = (seg_h.max() - ep) / ps
                row[f'ret{H}'] = ret
                row[f'mfe{H}'] = mfe
                row[f'mae{H}'] = mae
            rows.append(row)
    return rows


def null_baseline(epics, n_samples, H, seed=7):
    rng = np.random.default_rng(seed)
    rets = []
    pool = [(e, candles_cached(e, 60)) for e in epics]
    pool = [(e, d) for e, d in pool if d is not None]
    if not pool:
        return np.array([])
    for _ in range(n_samples):
        e, df = pool[rng.integers(len(pool))]
        c = df['close'].values; ps = pip(e); cst = cost(e)
        i = int(rng.integers(50, len(df) - H - 1))
        s = 1 if rng.random() > 0.5 else -1
        rets.append(s * (c[i + H] - c[i]) / ps - cst)
    return np.array(rets)


def summ(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x):5d} mean={x.mean():7.2f} med={np.median(x):7.2f} "
            f"%pos={100*(x>0).mean():4.1f}")


def main(min_sig, focus_h):
    sdf = pd.read_sql(
        """SELECT strategy FROM alert_history WHERE strategy IS NOT NULL
           GROUP BY strategy HAVING COUNT(*) >= %(m)s ORDER BY COUNT(*) DESC""",
        _dbm().get_engine(), params={'m': min_sig})
    strategies = list(sdf['strategy'])

    all_rows = []
    for st in strategies:
        all_rows.extend(signal_rows(st, min_sig))
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no rows"); return

    print(f"\n{'='*92}\nENTRY-QUALITY DIAGNOSTIC — {df['strategy'].nunique()} strategies, "
          f"{len(df)} signals, forward NET pips (after spread), 1h bars\n{'='*92}")

    print("\n[1] DIRECTIONAL SKILL vs NULL (random bar + random dir) — net return by horizon:")
    epics = sorted(df['epic'].unique())
    for H in HORIZONS:
        sig = df[f'ret{H}']
        nul = null_baseline(epics, min(8000, len(df) * 4), H)
        print(f"   H={H:3d}b  signals: {summ(sig)}   |   null: {summ(nul)}")

    H = focus_h
    print(f"\n[2] MFE / MAE at H={H}b (pips):")
    print(f"   mean MFE={df[f'mfe{H}'].mean():6.1f}  mean MAE={df[f'mae{H}'].mean():6.1f}  "
          f"MFE/MAE={df[f'mfe{H}'].mean()/max(df[f'mae{H}'].mean(),1e-9):.2f}  "
          f"%(MFE>=1.5*MAE)={100*(df[f'mfe{H}']>=1.5*df[f'mae{H}']).mean():.1f}")

    def slice_print(title, grp):
        print(f"\n[3] Net ret H={H}b by {title}:")
        g = df.groupby(grp)[f'ret{H}'].agg(['count', 'mean',
                                            lambda s: 100*(s>0).mean()])
        g.columns = ['n', 'mean_pips', 'pct_pos']
        g = g[g['n'] >= 20].sort_values('mean_pips', ascending=False)
        for idx, r in g.iterrows():
            print(f"   {str(idx):28s} n={int(r['n']):5d}  mean={r['mean_pips']:7.2f}  %pos={r['pct_pos']:4.1f}")

    df['session'] = pd.cut(df['hour'], [-1,6,12,16,21,24],
                           labels=['Asian_0-6','London_7-12','Overlap_13-16','NY_17-21','Late_22-23'])
    df['adx_bucket'] = pd.cut(df['adx'], [0,15,20,25,30,100],
                              labels=['<15','15-20','20-25','25-30','>30'])
    df['rsi_bucket'] = pd.cut(df['rsi'], [0,30,45,55,70,100],
                              labels=['<30','30-45','45-55','55-70','>70'])
    slice_print('session', 'session')
    slice_print('ADX regime', 'adx_bucket')
    slice_print('HTF EMA200 alignment', 'htf_aligned')
    slice_print('RSI bucket', 'rsi_bucket')
    slice_print('strategy', 'strategy')

    # ---- [4] Stacked context gate, IS/OOS time split ----
    # Gates chosen from diagnostic + prior independent evidence (session, RSI).
    rc = f'ret{H}'
    df = df.sort_values('ts').reset_index(drop=True)
    split = df['ts'].quantile(0.6)

    gates = {
        'ALL (no gate)': pd.Series(True, index=df.index),
        'ADX 18-28': df['adx'].between(18, 28),
        'session Asian+Overlap (0-16)': df['hour'].between(0, 16),
        'RSI 50-70': df['rsi'].between(50, 70),
        'STACK adx18-28 & hr0-16 & rsi50-70':
            df['adx'].between(18, 28) & df['hour'].between(0, 16) & df['rsi'].between(50, 70),
        'STACK + counter-EMA200':
            df['adx'].between(18, 28) & df['hour'].between(0, 16)
            & df['rsi'].between(50, 70) & (~df['htf_aligned']),
    }
    print(f"\n[4] STACKED CONTEXT GATE — net ret H={H}b, IS(<{split.date()}) / OOS(>=):")
    print(f"   {'gate':40s} {'IS n/mean/%pos':>22s}   {'OOS n/mean/%pos':>22s}")
    for name, mask in gates.items():
        sub = df[mask]
        isd = sub[sub['ts'] < split][rc]; oos = sub[sub['ts'] >= split][rc]
        def f(s): return f"{len(s):5d}/{s.mean():7.2f}/{100*(s>0).mean():4.1f}" if len(s) else "    0/   -  /  - "
        print(f"   {name:40s} {f(isd):>22s}   {f(oos):>22s}")


def debias(min_sig, H, split_frac, min_bucket_n, min_is_edge):
    """Select gate buckets on IS ONLY, lock, then evaluate on OOS (untouched)."""
    sdf = pd.read_sql(
        """SELECT strategy FROM alert_history WHERE strategy IS NOT NULL
           GROUP BY strategy HAVING COUNT(*) >= %(m)s ORDER BY COUNT(*) DESC""",
        _dbm().get_engine(), params={'m': min_sig})
    rows = []
    for st in list(sdf['strategy']):
        rows.extend(signal_rows(st, min_sig))
    df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    rc = f'ret{H}'
    split = df['ts'].quantile(split_frac)
    IS = df[df['ts'] < split]; OOS = df[df['ts'] >= split]

    df['adx_b'] = pd.cut(df['adx'], [0,10,15,18,20,22,25,28,30,35,100]).astype(str)
    df['rsi_b'] = pd.cut(df['rsi'], [0,30,40,45,50,55,60,70,100]).astype(str)
    df['hr_b'] = df['hour'].astype(str)
    IS = df[df['ts'] < split]; OOS = df[df['ts'] >= split]

    def pick(col):
        g = IS.groupby(col)[rc].agg(['count', 'mean'])
        keep = set(g[(g['count'] >= min_bucket_n) & (g['mean'] > min_is_edge)].index)
        return keep

    print(f"\n{'='*92}\nDE-BIASED GATE — buckets chosen on IS(<{split.date()}) ONLY, tested on OOS\n"
          f"  (min bucket n={min_bucket_n}, min IS edge={min_is_edge} pips, H={H}b)\n{'='*92}")
    sel = {}
    for col, name in [('adx_b', 'ADX'), ('rsi_b', 'RSI'), ('hr_b', 'hour')]:
        sel[col] = pick(col)
        print(f"  IS-selected {name:5s} buckets: {sorted(sel[col])}")

    def rep(mask_df, label):
        isd = mask_df[mask_df['ts'] < split][rc]; oos = mask_df[mask_df['ts'] >= split][rc]
        def f(s): return f"n={len(s):5d} mean={s.mean():7.2f} %pos={100*(s>0).mean():4.1f}" if len(s) else "n=0"
        print(f"   {label:34s} IS[{f(isd)}]  OOS[{f(oos)}]")

    print("\nLOCKED single-feature gates (IS-selected buckets):")
    m_adx = df['adx_b'].isin(sel['adx_b'])
    m_rsi = df['rsi_b'].isin(sel['rsi_b'])
    m_hr = df['hr_b'].isin(sel['hr_b'])
    rep(df, 'ungated (reference)')
    rep(df[m_adx], 'ADX-locked')
    rep(df[m_rsi], 'RSI-locked')
    rep(df[m_hr], 'hour-locked')
    print("\nLOCKED STACKED gate:")
    stack = df[m_adx & m_rsi & m_hr]
    rep(stack, 'ADX & RSI & hour locked')
    print("\nPer-strategy OOS of stacked locked gate (n>=15 OOS):")
    o = stack[stack['ts'] >= split]
    g = o.groupby('strategy')[rc].agg(['count', 'mean', lambda s: 100*(s>0).mean()])
    g.columns = ['n', 'mean', 'pct_pos']
    for idx, r in g[g['n'] >= 15].sort_values('mean', ascending=False).iterrows():
        print(f"   {str(idx):26s} n={int(r['n']):4d}  mean={r['mean']:7.2f}  %pos={r['pct_pos']:4.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-sig', type=int, default=25)
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--debias', action='store_true')
    ap.add_argument('--split', type=float, default=0.55)
    ap.add_argument('--min-bucket-n', type=int, default=30)
    ap.add_argument('--min-is-edge', type=float, default=0.0)
    a = ap.parse_args()
    if a.debias:
        debias(a.min_sig, a.horizon, a.split, a.min_bucket_n, a.min_is_edge)
    else:
        main(a.min_sig, a.horizon)
