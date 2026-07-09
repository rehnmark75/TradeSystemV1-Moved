#!/usr/bin/env python3
"""
EXPERIMENTAL (Jul 6 2026) — EURUSD strategy lab: exit-model bench + entry search.

Methodology (house rules):
  - de-bias: every entry set is benchmarked against time-matched random entries
    with the SAME exit model; an exit that "makes money" on random entries is an
    artifact (Neo Cloud lesson) and is flagged, not celebrated.
  - all open positions are marked to market and force-closed at horizon/end —
    no losers parked unrealized.
  - costs: 2.0 pips round-trip (reference_ig_realized_spread_jun15).
  - signal computed at bar close, entry at NEXT bar open (no lookahead).
  - splits: IS 2020-01..2023-12 | OOS-A 2024-01..2025-03 | OOS-B 2025-04..2026-06.

Subcommands:
  exits     phase 1: exit-family bench on {random, momentum, meanrev} entry sets
  entries   phase 2: broad entry-condition screen, forward net returns vs null
  combo     phase 3: full single-position sim of surviving entries x exits

Run:
  docker exec task-worker python /app/forex_scanner/eurusd_lab.py exits
"""
from __future__ import annotations
import sys, argparse, itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

EPIC = 'CS.D.EURUSD.CEEM.IP'
PIP = 0.0001
COST = 2.0          # round-trip pips; override with --cost (user: EURUSD avg 0.6)
SEED = 7

SPLITS = {
    'IS':    ('2020-01-01', '2023-12-31'),
    'OOS-A': ('2024-01-01', '2025-03-31'),
    'OOS-B': ('2025-04-01', '2026-06-30'),
}


# ---------------------------------------------------------------- data
def _engine():
    from core.database import DatabaseManager
    import config
    return DatabaseManager(config.DATABASE_URL).get_engine()


_CACHE: dict = {}


def load(tf: int) -> pd.DataFrame:
    if tf in _CACHE:
        return _CACHE[tf]
    df = pd.read_sql(
        """SELECT start_time, open, high, low, close FROM ig_candles_backtest
           WHERE epic=%(e)s AND timeframe=%(tf)s AND start_time>='2019-12-01'
           ORDER BY start_time""",
        _engine(), params={'e': EPIC, 'tf': tf})
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.drop_duplicates('start_time').reset_index(drop=True)
    _CACHE[tf] = df
    return df


# ---------------------------------------------------------------- indicators
def wilder(arr, period):
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = np.nanmean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def add_ind(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    df = df.copy()
    df['atr'] = wilder(tr, 14)

    up = h - np.roll(h, 1); up[0] = 0
    dn = np.roll(l, 1) - l; dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_di = wilder(tr, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        pdi = 100 * wilder(pdm, 14) / atr_di
        mdi = 100 * wilder(mdm, 14) / atr_di
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    df['adx'] = wilder(np.nan_to_num(dx), 14)

    cs = pd.Series(c)
    for n in (9, 21, 50, 200):
        df[f'ema{n}'] = cs.ewm(span=n, adjust=False).mean().values

    delta = np.diff(c, prepend=c[0])
    gain = wilder(np.where(delta > 0, delta, 0.0), 14)
    loss = wilder(np.where(delta < 0, -delta, 0.0), 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        df['rsi'] = 100 - 100 / (1 + gain / loss)

    # bollinger 20,2 and z-score of close vs ema50 in ATR units
    m = cs.rolling(20).mean(); sd = cs.rolling(20).std()
    df['bb_up'] = (m + 2 * sd).values
    df['bb_dn'] = (m - 2 * sd).values
    with np.errstate(invalid='ignore', divide='ignore'):
        df['ext_atr'] = (c - df['ema50'].values) / df['atr'].values
    df['don_hi'] = pd.Series(h).rolling(20).max().shift(1).values
    df['don_lo'] = pd.Series(l).rolling(20).min().shift(1).values
    df['don_hi55'] = pd.Series(h).rolling(55).max().shift(1).values
    df['don_lo55'] = pd.Series(l).rolling(55).min().shift(1).values
    df['body_atr'] = np.abs(c - df['open'].values) / np.where(df['atr'].values == 0, np.nan, df['atr'].values)
    df['hour'] = df['start_time'].dt.hour
    df['dow'] = df['start_time'].dt.dayofweek
    return df


def htf_context(df5: pd.DataFrame) -> pd.DataFrame:
    """Merge 1h indicator context onto 5m frame (as-of previous COMPLETED 1h bar)."""
    d1 = add_ind(load(60))
    d1 = d1[['start_time', 'adx', 'ema21', 'ema50', 'ema200', 'rsi', 'close', 'atr']].copy()
    d1.columns = ['start_time'] + [f'h1_{c}' for c in d1.columns[1:]]
    # a 1h bar stamped T covers T..T+1h and completes at T+1h
    d1['avail'] = d1['start_time'] + pd.Timedelta(hours=1)
    out = pd.merge_asof(df5.sort_values('start_time'), d1.drop(columns=['start_time']).sort_values('avail'),
                        left_on='start_time', right_on='avail', direction='backward')
    # prior-day high/low from completed daily bars
    day = df5.set_index('start_time').resample('1D').agg(
        {'high': 'max', 'low': 'min'}).dropna()
    day = day.shift(1).rename(columns={'high': 'pdh', 'low': 'pdl'}).reset_index()
    day['d'] = day['start_time'].dt.date
    out['d'] = out['start_time'].dt.date
    out = out.merge(day[['d', 'pdh', 'pdl']], on='d', how='left').drop(columns=['d'])
    return out


def prep() -> pd.DataFrame:
    df = add_ind(load(5))
    df = htf_context(df)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- entry sets
def entries_random(df, n=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    lo, hi = 300, len(df) - 2000
    idx = np.sort(rng.choice(np.arange(lo, hi), size=n, replace=False))
    dirs = np.where(rng.random(n) > 0.5, 1, -1)
    return idx, dirs


def entries_momentum(df):
    """1h trend up + 5m 20-bar breakout (and mirrored short)."""
    c = df['close'].values
    up = (df['h1_ema21'].values > df['h1_ema50'].values) & (c > df['don_hi'].values)
    dn = (df['h1_ema21'].values < df['h1_ema50'].values) & (c < df['don_lo'].values)
    return _collapse(df, up, dn, gap=12)


def entries_meanrev(df):
    """Fade >2 ATR extension from 5m EMA50 when 1h ADX < 30."""
    ext = df['ext_atr'].values
    calm = df['h1_adx'].values < 30
    up = (ext < -2.0) & calm          # buy the dip
    dn = (ext > 2.0) & calm           # sell the rip
    return _collapse(df, up, dn, gap=12)


def _collapse(df, up, dn, gap=12):
    """Turn boolean signal arrays into deduped (idx, dir) with min gap bars."""
    idx_list, dir_list = [], []
    last = -10**9
    sig = np.where(up, 1, np.where(dn, -1, 0))
    nz = np.nonzero(sig)[0]
    for i in nz:
        if i - last >= gap and 300 <= i < len(df) - 2000:
            idx_list.append(i); dir_list.append(sig[i])
            last = i
    return np.array(idx_list), np.array(dir_list)


# ---------------------------------------------------------------- exit engine
@dataclass
class ExitSpec:
    name: str
    kind: str            # 'bracket','atr_bracket','time','trail','be_trail','sl_time','sl_signal'
    sl: float = 0.0      # pips or ATR-mult depending on kind
    tp: float = 0.0
    time_bars: int = 0
    trail_mult: float = 0.0
    arm: float = 0.0     # pips profit to arm trail / breakeven
    max_bars: int = 2000  # hard cap for everything (mark-to-market close)


def sim_exit(df, idx, dirs, spec: ExitSpec):
    """Per-entry independent sim. Entry at next bar OPEN. Returns net pips array."""
    o = df['open'].values; h = df['high'].values; l = df['low'].values
    c = df['close'].values; atr = df['atr'].values
    ext = df['ext_atr'].values
    n = len(df)
    outs = np.empty(len(idx)); bars_held = np.empty(len(idx))
    for k in range(len(idx)):
        i0 = idx[k] + 1                      # entry bar
        s = dirs[k]
        ep = o[i0]
        if spec.kind == 'bracket':
            sl_px = ep - s * spec.sl * PIP; tp_px = ep + s * spec.tp * PIP
        elif spec.kind == 'atr_bracket':
            a = atr[idx[k]]
            sl_px = ep - s * spec.sl * a; tp_px = ep + s * spec.tp * a
        else:
            a = atr[idx[k]]
            sl_px = ep - s * spec.sl * PIP if spec.sl > 0 else None
            tp_px = None
        peak = ep
        be_armed = False
        res = None; jend = min(n - 1, i0 + spec.max_bars)
        for j in range(i0, jend):
            if s > 0:
                hit_sl = sl_px is not None and l[j] <= sl_px
                hit_tp = tp_px is not None and h[j] >= tp_px
            else:
                hit_sl = sl_px is not None and h[j] >= sl_px
                hit_tp = tp_px is not None and l[j] <= tp_px
            if hit_sl:                        # conservative: SL first
                res = (sl_px, j); break
            if hit_tp:
                res = (tp_px, j); break
            cur = s * (c[j] - ep) / PIP
            if spec.kind in ('trail', 'be_trail'):
                peak = max(peak, h[j]) if s > 0 else min(peak, l[j])
                peak_pips = s * (peak - ep) / PIP
                if spec.kind == 'be_trail' and not be_armed and peak_pips >= spec.arm:
                    be_px = ep + s * 1.0 * PIP
                    sl_px = max(sl_px, be_px) if (sl_px is not None and s > 0) else \
                            (min(sl_px, be_px) if sl_px is not None else be_px)
                    be_armed = True
                if peak_pips >= spec.arm and spec.trail_mult > 0:
                    give = spec.trail_mult * atr[j] / PIP
                    trail_px = peak - s * give * PIP
                    if sl_px is None:
                        sl_px = trail_px
                    else:
                        sl_px = max(sl_px, trail_px) if s > 0 else min(sl_px, trail_px)
            if spec.kind in ('time', 'sl_time') and (j - i0 + 1) >= spec.time_bars:
                res = (c[j], j); break
            if spec.kind == 'sl_signal':
                # signal-exit: close when extension reverts to mean (ext crosses 0
                # against us) — generic "thesis done" close
                if (s > 0 and ext[j] >= 0) or (s < 0 and ext[j] <= 0):
                    if j - i0 >= 2:
                        res = (c[j], j); break
        if res is None:
            res = (c[jend], jend)
        outs[k] = s * (res[0] - ep) / PIP - COST
        bars_held[k] = res[1] - i0 + 1
    return outs, bars_held


EXIT_SPECS = [
    ExitSpec('bracket 10/10', 'bracket', sl=10, tp=10),
    ExitSpec('bracket 10/20', 'bracket', sl=10, tp=20),
    ExitSpec('bracket 15/15', 'bracket', sl=15, tp=15),
    ExitSpec('bracket 15/30', 'bracket', sl=15, tp=30),
    ExitSpec('bracket 25/25', 'bracket', sl=25, tp=25),
    ExitSpec('bracket 25/50', 'bracket', sl=25, tp=50),
    ExitSpec('bracket 30/15 (scalp-ish)', 'bracket', sl=30, tp=15),
    ExitSpec('atr 1.5/1.5', 'atr_bracket', sl=1.5, tp=1.5),
    ExitSpec('atr 2/4', 'atr_bracket', sl=2.0, tp=4.0),
    ExitSpec('atr 3/6', 'atr_bracket', sl=3.0, tp=6.0),
    ExitSpec('time 12b (1h)', 'time', time_bars=12),
    ExitSpec('time 48b (4h)', 'time', time_bars=48),
    ExitSpec('time 144b (12h)', 'time', time_bars=144),
    ExitSpec('time 288b (1d)', 'time', time_bars=288),
    ExitSpec('sl15 + time 48b', 'sl_time', sl=15, time_bars=48),
    ExitSpec('sl20 + time 144b', 'sl_time', sl=20, time_bars=144),
    ExitSpec('sl25 + time 288b', 'sl_time', sl=25, time_bars=288),
    ExitSpec('trail atr2 arm5 sl15', 'trail', sl=15, trail_mult=2.0, arm=5),
    ExitSpec('trail atr3 arm8 sl20', 'trail', sl=20, trail_mult=3.0, arm=8),
    ExitSpec('be+trail atr2 arm8 sl15', 'be_trail', sl=15, trail_mult=2.0, arm=8),
    ExitSpec('sl15 + revert-signal exit', 'sl_signal', sl=15),
    ExitSpec('sl25 + revert-signal exit', 'sl_signal', sl=25),
]


# ---------------------------------------------------------------- reporting
def stats(net):
    net = np.asarray(net, float)
    if len(net) == 0:
        return dict(n=0, pf=np.nan, exp=np.nan, pos=np.nan)
    gw = net[net > 0].sum(); gl = -net[net < 0].sum()
    return dict(n=len(net), pf=(gw / gl if gl > 0 else np.inf),
                exp=net.mean(), pos=100 * (net > 0).mean())


def split_mask(df, idx, split):
    lo, hi = SPLITS[split]
    ts = df['start_time'].values[idx]
    return (ts >= np.datetime64(lo)) & (ts <= np.datetime64(hi))


def run_exits():
    df = prep()
    sets = {
        'RANDOM': entries_random(df),
        'MOM': entries_momentum(df),
        'MEANREV': entries_meanrev(df),
    }
    print(f"data: {len(df)} 5m bars {df['start_time'].iloc[0]} .. {df['start_time'].iloc[-1]}")
    for nm, (idx, dirs) in sets.items():
        print(f"entry set {nm}: n={len(idx)}")
    hdr = f"{'exit':32s} {'set':8s}" + "".join(
        f" | {sp:>5s} n / pf / exp / %pos" for sp in SPLITS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for spec in EXIT_SPECS:
        for nm, (idx, dirs) in sets.items():
            net, bars = sim_exit(df, idx, dirs, spec)
            line = f"{spec.name:32s} {nm:8s}"
            for sp in SPLITS:
                m = split_mask(df, idx, sp)
                st = stats(net[m])
                line += (f" | {st['n']:5d} {st['pf']:5.2f} {st['exp']:6.2f} {st['pos']:4.1f}"
                         if st['n'] else " |     0     -      -    -")
            line += f"  avg_bars={np.mean(bars):.0f}"
            print(line, flush=True)


# ---------------------------------------------------------------- entry screen
def condition_library(df):
    """Return list of (name, up_bool, dn_bool). All computed at bar close."""
    c = df['close'].values; o = df['open'].values
    h = df['high'].values; l = df['low'].values
    rsi = df['rsi'].values; ext = df['ext_atr'].values
    adx1h = df['h1_adx'].values; hour = df['hour'].values
    e9, e21, e50 = df['ema9'].values, df['ema21'].values, df['ema50'].values
    h1e21, h1e50, h1e200 = df['h1_ema21'].values, df['h1_ema50'].values, df['h1_ema200'].values
    h1rsi = df['h1_rsi'].values
    bbu, bbd = df['bb_up'].values, df['bb_dn'].values
    body = df['body_atr'].values
    pdh, pdl = df['pdh'].values, df['pdl'].values
    don_hi55, don_lo55 = df['don_hi55'].values, df['don_lo55'].values
    london = (hour >= 6) & (hour <= 13)
    lib = []

    def add(name, up, dn):
        lib.append((name, np.nan_to_num(up).astype(bool), np.nan_to_num(dn).astype(bool)))

    add('rsi5 <25/>75 fade', rsi < 25, rsi > 75)
    add('rsi5 <30/>70 fade', rsi < 30, rsi > 70)
    add('rsi1h <30/>70 fade', h1rsi < 30, h1rsi > 70)
    add('ext>2atr fade', ext < -2.0, ext > 2.0)
    add('ext>2.5atr fade', ext < -2.5, ext > 2.5)
    add('ext>2atr fade calm1h', (ext < -2.0) & (adx1h < 25), (ext > 2.0) & (adx1h < 25))
    add('bb2 close-outside fade', c < bbd, c > bbu)
    add('bb2 fade + rsi conf', (c < bbd) & (rsi < 30), (c > bbu) & (rsi > 70))
    add('impulse2.2 fade', (body > 2.2) & (c < o), (body > 2.2) & (c > o))
    add('don20 breakout w/1h trend',
        (c > df['don_hi'].values) & (h1e21 > h1e50),
        (c < df['don_lo'].values) & (h1e21 < h1e50))
    add('don55 breakout w/1h trend',
        (c > don_hi55) & (h1e21 > h1e50),
        (c < don_lo55) & (h1e21 < h1e50))
    add('ema stack pullback',    # trend up, pullback touched ema21, close back above ema9
        (e9 > e21) & (e21 > e50) & (l <= e21) & (c > e9),
        (e9 < e21) & (e21 < e50) & (h >= e21) & (c < e9))
    add('ema stack pullback +1h align',
        (e9 > e21) & (e21 > e50) & (l <= e21) & (c > e9) & (h1e21 > h1e50),
        (e9 < e21) & (e21 < e50) & (h >= e21) & (c < e9) & (h1e21 < h1e50))
    add('pdh/pdl sweep-reclaim',    # wick beyond prior-day level, close back inside
        (l < pdl) & (c > pdl), (h > pdh) & (c < pdh))
    add('pdh/pdl sweep-reclaim london',
        (l < pdl) & (c > pdl) & london, (h > pdh) & (c < pdh) & london)
    add('pdh/pdl breakout close',
        (c > pdh) & (o <= pdh), (c < pdl) & (o >= pdl))
    add('1h trend + 5m ext pullback',   # buy dips in 1h uptrend
        (h1e21 > h1e50) & (c > h1e200) & (ext < -1.5),
        (h1e21 < h1e50) & (c < h1e200) & (ext > 1.5))
    add('1h trend + rsi5 pullback',
        (h1e21 > h1e50) & (rsi < 35), (h1e21 < h1e50) & (rsi > 65))
    add('adx1h 15-25 + ext fade (survivor)',
        (adx1h >= 15) & (adx1h <= 25) & (ext < -2.0),
        (adx1h >= 15) & (adx1h <= 25) & (ext > 2.0))
    add('london bb fade', (c < bbd) & london, (c > bbu) & london)
    add('london ema-stack pullback',
        (e9 > e21) & (e21 > e50) & (l <= e21) & (c > e9) & london,
        (e9 < e21) & (e21 < e50) & (h >= e21) & (c < e9) & london)
    add('3-dn-closes in 1h-up (dip)',
        (h1e21 > h1e50) & (c < o) & (np.roll(c < o, 1)) & (np.roll(c < o, 2)),
        (h1e21 < h1e50) & (c > o) & (np.roll(c > o, 1)) & (np.roll(c > o, 2)))
    return lib


HORIZONS = [12, 48, 144, 288]     # 1h, 4h, 12h, 1d in 5m bars


def fwd_net(df, idx, dirs, H):
    o = df['open'].values; c = df['close'].values
    i0 = idx + 1
    ep = o[i0]
    xp = c[np.minimum(i0 + H, len(df) - 1)]
    return dirs * (xp - ep) / PIP - COST


def run_entries():
    df = prep()
    lib = condition_library(df)
    rng = np.random.default_rng(SEED)
    print(f"data: {len(df)} bars | {len(lib)} conditions | horizons {HORIZONS} (5m bars)")
    hdr = (f"{'condition':38s} {'split':6s} {'n':>6s}" +
           "".join(f" | H{H:<3d} exp  t  %pos xs" for H in HORIZONS))
    print(hdr); print("-" * len(hdr))
    for name, up, dn in lib:
        idx, dirs = _collapse(df, up, dn, gap=12)
        if len(idx) < 60:
            print(f"{name:38s} SKIP n={len(idx)}")
            continue
        for sp in SPLITS:
            m = split_mask(df, idx, sp)
            if m.sum() < 30:
                continue
            i_s, d_s = idx[m], dirs[m]
            # time-matched null: same bars, random direction (controls drift)
            d_null = np.where(rng.random(len(i_s)) > 0.5, 1, -1)
            line = f"{name:38s} {sp:6s} {len(i_s):6d}"
            for H in HORIZONS:
                r = fwd_net(df, i_s, d_s, H)
                rn = fwd_net(df, i_s, d_null, H)
                t = r.mean() / (r.std() / np.sqrt(len(r)) + 1e-9)
                xs = r.mean() - rn.mean()
                line += f" | {r.mean():6.2f} {t:4.1f} {100*(r>0).mean():4.1f} {xs:6.2f}"
            print(line, flush=True)


# ---------------------------------------------------------------- refine
def refine_library(df):
    """Variants of the two surviving families: trend+rsi5 pullback, rsi1h fade."""
    c = df['close'].values
    rsi = df['rsi'].values; ext = df['ext_atr'].values
    adx1h = df['h1_adx'].values; hour = df['hour'].values
    h1e21, h1e50, h1e200 = df['h1_ema21'].values, df['h1_ema50'].values, df['h1_ema200'].values
    h1rsi = df['h1_rsi'].values
    up_t = h1e21 > h1e50; dn_t = h1e21 < h1e50
    eu = (hour >= 6) & (hour <= 16)
    lib = []

    def add(name, up, dn):
        lib.append((name, np.nan_to_num(up).astype(bool), np.nan_to_num(dn).astype(bool)))

    for x in (30, 35, 40):
        add(f'A rsi5<{x} in 1h-trend', up_t & (rsi < x), dn_t & (rsi > 100 - x))
    add('A rsi5<35 +ema200 align', up_t & (rsi < 35) & (c > h1e200),
        dn_t & (rsi > 65) & (c < h1e200))
    add('A rsi5<35 +adx1h>=18', up_t & (rsi < 35) & (adx1h >= 18),
        dn_t & (rsi > 65) & (adx1h >= 18))
    add('A rsi5<35 +adx1h<18', up_t & (rsi < 35) & (adx1h < 18),
        dn_t & (rsi > 65) & (adx1h < 18))
    add('A rsi5<35 hours6-16', up_t & (rsi < 35) & eu, dn_t & (rsi > 65) & eu)
    add('A rsi5<35 +ext<-1', up_t & (rsi < 35) & (ext < -1.0),
        dn_t & (rsi > 65) & (ext > 1.0))
    add('A rsi5<35 +h1rsi<50', up_t & (rsi < 35) & (h1rsi < 50),
        dn_t & (rsi > 65) & (h1rsi > 50))
    add('A rsi5<35 +h1rsi>50 (fresh)', up_t & (rsi < 35) & (h1rsi > 50),
        dn_t & (rsi > 65) & (h1rsi < 50))
    for x in (25, 30, 35):
        add(f'B rsi1h<{x} fade', h1rsi < x, h1rsi > 100 - x)
    add('B rsi1h<30 +rsi5<30 conf', (h1rsi < 30) & (rsi < 30),
        (h1rsi > 70) & (rsi > 70))
    add('B rsi1h<30 +calm adx<25', (h1rsi < 30) & (adx1h < 25),
        (h1rsi > 70) & (adx1h < 25))
    add('B rsi1h<30 hours6-16', (h1rsi < 30) & eu, (h1rsi > 70) & eu)
    add('A+B rsi5<35 & rsi1h<40 1h-up', up_t & (rsi < 35) & (h1rsi < 40),
        dn_t & (rsi > 65) & (h1rsi > 60))
    return lib


def run_refine(gap=24):
    df = prep()
    lib = refine_library(df)
    rng = np.random.default_rng(SEED)
    HS = [144, 288]
    print(f"gap={gap} bars | horizons {HS}")
    hdr = (f"{'condition':34s} {'split':6s} {'n':>5s} {'nB':>5s}" +
           "".join(f" | H{H}: exp   xs  xsB  xsS" for H in HS))
    print(hdr); print("-" * len(hdr))
    for name, up, dn in lib:
        idx, dirs = _collapse(df, up, dn, gap=gap)
        if len(idx) < 60:
            print(f"{name:34s} SKIP n={len(idx)}")
            continue
        for sp in SPLITS:
            m = split_mask(df, idx, sp)
            if m.sum() < 30:
                continue
            i_s, d_s = idx[m], dirs[m]
            d_null = np.where(rng.random(len(i_s)) > 0.5, 1, -1)
            line = f"{name:34s} {sp:6s} {len(i_s):5d} {(d_s>0).sum():5d}"
            for H in HS:
                r = fwd_net(df, i_s, d_s, H)
                rn = fwd_net(df, i_s, d_null, H)
                xs = r.mean() - rn.mean()
                # per-direction excess: signal ret vs null ret on same bars
                mb, ms = d_s > 0, d_s < 0
                xsb = (r[mb].mean() - rn[mb].mean()) if mb.sum() > 10 else np.nan
                xss = (r[ms].mean() - rn[ms].mean()) if ms.sum() > 10 else np.nan
                line += f" | {r.mean():6.2f} {xs:5.2f} {xsb:5.2f} {xss:5.2f}"
            print(line, flush=True)


# ---------------------------------------------------------------- combo sim
def sim_sequential(df, up, dn, spec: ExitSpec, cooldown=6):
    """Single-position, no overlap: skip signals while a trade is open."""
    idx_all, dirs_all = _collapse(df, up, dn, gap=1)
    o = df['open'].values
    trades_net, trades_ts, trades_dir, trades_bars = [], [], [], []
    busy_until = -1
    for k in range(len(idx_all)):
        i = idx_all[k]
        if i <= busy_until:
            continue
        net, bars = sim_exit(df, np.array([i]), np.array([dirs_all[k]]), spec)
        trades_net.append(net[0]); trades_bars.append(bars[0])
        trades_ts.append(df['start_time'].values[i]); trades_dir.append(dirs_all[k])
        busy_until = i + int(bars[0]) + cooldown
    return (np.array(trades_net), np.array(trades_ts),
            np.array(trades_dir), np.array(trades_bars))


def run_combo():
    df = prep()
    c = df['close'].values
    rsi = df['rsi'].values
    h1e21, h1e50, h1e200 = df['h1_ema21'].values, df['h1_ema50'].values, df['h1_ema200'].values
    h1rsi = df['h1_rsi'].values
    up_t = h1e21 > h1e50; dn_t = h1e21 < h1e50

    a_up = up_t & (rsi < 35); a_dn = dn_t & (rsi > 65)
    b_up = (h1rsi < 30) & (rsi < 30); b_dn = (h1rsi > 70) & (rsi > 70)
    zeros = np.zeros(len(df), bool)
    entries = {
        'A rsi5<35 1h-trend': (a_up, a_dn),
        'A BUY-only': (a_up, zeros),
        'B dual-oversold': (b_up, b_dn),
        'A|B union': (a_up | b_up, a_dn | b_dn),
    }
    exits = [
        ExitSpec('bracket 25/50', 'bracket', sl=25, tp=50),
        ExitSpec('bracket 25/25', 'bracket', sl=25, tp=25),
        ExitSpec('bracket 40/40', 'bracket', sl=40, tp=40),
        ExitSpec('bracket 30/60', 'bracket', sl=30, tp=60),
        ExitSpec('sl25 + time 288b', 'sl_time', sl=25, time_bars=288),
        ExitSpec('sl30 + time 144b', 'sl_time', sl=30, time_bars=144),
        ExitSpec('time 288b', 'time', time_bars=288),
    ]
    for enm, (up, dn) in entries.items():
        up = np.nan_to_num(up).astype(bool); dn = np.nan_to_num(dn).astype(bool)
        for spec in exits:
            net, ts, dirs, bars = sim_sequential(df, up, dn, spec)
            line = f"{enm:26s} x {spec.name:18s}"
            for sp in SPLITS:
                lo, hi = SPLITS[sp]
                m = (ts >= np.datetime64(lo)) & (ts <= np.datetime64(hi))
                st = stats(net[m])
                months = max(1, (np.datetime64(hi) - np.datetime64(lo)) / np.timedelta64(30, 'D'))
                tpm = st['n'] / float(months)
                line += (f" | {sp} n={st['n']:4d} pf={st['pf']:4.2f} "
                         f"exp={st['exp']:5.2f} t/mo={tpm:4.1f}")
            print(line, flush=True)


# ---------------------------------------------------------------- limit-entry sim
def sim_limit_sequential(df, up, dn, k_atr, exit_mode, sl_pips=25.0,
                         max_bars=288, ttl=12, cooldown=6):
    """Mean-reversion limit entry: on signal bar i, rest a limit k_atr*ATR
    beyond the close; fill if touched within ttl bars. Exit modes:
      'time'   : SL + time cap
      'revert' : TP on touch of 5m EMA50 (reversion target) + SL + time cap
      'bracket': SL + 2x TP fixed
    Single position, fills at limit or better. Returns per-trade arrays.
    """
    idx_all, dirs_all = _collapse(df, np.nan_to_num(up).astype(bool),
                                  np.nan_to_num(dn).astype(bool), gap=1)
    o = df['open'].values; h = df['high'].values; l = df['low'].values
    c = df['close'].values; atr = df['atr'].values; e50 = df['ema50'].values
    n = len(df)
    net_l, ts_l, held_l, filled_ct, sig_ct = [], [], [], 0, 0
    busy_until = -1
    for kk in range(len(idx_all)):
        i = idx_all[kk]
        if i <= busy_until:
            continue
        sig_ct += 1
        s = dirs_all[kk]
        lim = c[i] - s * k_atr * atr[i]
        # ---- fill window
        j0 = None; ep = None
        for j in range(i + 1, min(i + 1 + ttl, n - max_bars - 2)):
            if s > 0:
                if o[j] <= lim:
                    j0, ep = j, o[j]; break
                if l[j] <= lim:
                    j0, ep = j, lim; break
            else:
                if o[j] >= lim:
                    j0, ep = j, o[j]; break
                if h[j] >= lim:
                    j0, ep = j, lim; break
        if j0 is None:
            busy_until = i + ttl        # order expired
            continue
        filled_ct += 1
        sl_px = ep - s * sl_pips * PIP
        tp_px = ep + s * 2 * sl_pips * PIP if exit_mode == 'bracket' else None
        res = None
        jend = min(n - 1, j0 + max_bars)
        for j in range(j0, jend):
            hit_sl = (l[j] <= sl_px) if s > 0 else (h[j] >= sl_px)
            if hit_sl:
                res = (sl_px, j); break
            if exit_mode == 'bracket':
                hit_tp = (h[j] >= tp_px) if s > 0 else (l[j] <= tp_px)
                if hit_tp:
                    res = (tp_px, j); break
            if exit_mode == 'revert' and j > j0:
                if s > 0 and h[j] >= e50[j]:
                    res = (max(o[j], e50[j]) if o[j] >= e50[j] else e50[j], j); break
                if s < 0 and l[j] <= e50[j]:
                    res = (min(o[j], e50[j]) if o[j] <= e50[j] else e50[j], j); break
        if res is None:
            res = (c[jend], jend)
        net_l.append(s * (res[0] - ep) / PIP - COST)
        ts_l.append(df['start_time'].values[i])
        held_l.append(res[1] - j0 + 1)
        busy_until = res[1] + cooldown
    return (np.array(net_l), np.array(ts_l), np.array(held_l),
            filled_ct, sig_ct)


def run_combo2():
    df = prep()
    rsi = df['rsi'].values; h1rsi = df['h1_rsi'].values
    up_t = df['h1_ema21'].values > df['h1_ema50'].values
    dn_t = df['h1_ema21'].values < df['h1_ema50'].values
    zeros = np.zeros(len(df), bool)
    a_up = up_t & (rsi < 35); a_dn = dn_t & (rsi > 65)
    b_up = (h1rsi < 30) & (rsi < 30); b_dn = (h1rsi > 70) & (rsi > 70)
    entries = {
        'A rsi5<35 1h-trend': (a_up, a_dn),
        'A BUY-only': (a_up, zeros),
        'B dual-oversold': (b_up, b_dn),
    }
    grids = [(k, em, sl) for k in (0.5, 1.0, 1.5)
             for em in ('time', 'revert', 'bracket') for sl in (25.0,)]
    for enm, (up, dn) in entries.items():
        for k, em, sl in grids:
            net, ts, held, fc, sc = sim_limit_sequential(df, up, dn, k, em, sl_pips=sl)
            if len(net) == 0:
                continue
            line = f"{enm:20s} k={k:.1f} {em:8s}"
            for sp in SPLITS:
                lo, hi = SPLITS[sp]
                m = (ts >= np.datetime64(lo)) & (ts <= np.datetime64(hi))
                st = stats(net[m])
                line += (f" | {sp} n={st['n']:4d} pf={st['pf']:4.2f} exp={st['exp']:5.2f}"
                         if st['n'] else f" | {sp} n=   0")
            line += f"  fill={100*fc/max(1,sc):.0f}% hold={np.mean(held):.0f}b"
            print(line, flush=True)


# ---------------------------------------------------------------- overlay
def run_overlay(strategy='SMC_SIMPLE', lookback=12):
    """Grade live alert_history signals by the entry-A overlay state:
    trend_ok  = signal direction matches 1h ema21/ema50 trend
    pullback  = 5m RSI touched <35 (BUY) / >65 (SELL) within last `lookback` bars
    Report forward net returns + bracket sims per overlay cell."""
    df = prep()
    sig_df = pd.read_sql(
        """SELECT alert_timestamp, signal_type FROM alert_history
           WHERE epic=%(e)s AND strategy=%(s)s ORDER BY alert_timestamp""",
        _engine(), params={'e': EPIC, 's': strategy})
    ts_arr = df['start_time'].values
    rsi = df['rsi'].values
    up_t = df['h1_ema21'].values > df['h1_ema50'].values
    rsi_min = pd.Series(rsi).rolling(lookback).min().values
    rsi_max = pd.Series(rsi).rolling(lookback).max().values

    rows = []
    for _, r in sig_df.iterrows():
        t = pd.Timestamp(r['alert_timestamp'])
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        i = int(np.searchsorted(ts_arr, np.datetime64(t))) - 1  # last completed bar
        if i < 300 or i >= len(df) - 300:
            continue
        d = 1 if str(r['signal_type']).upper() in ('BUY', 'BULL', 'LONG') else -1
        trend_ok = bool(up_t[i]) if d > 0 else (not bool(up_t[i]))
        pullback = (rsi_min[i] < 35) if d > 0 else (rsi_max[i] > 65)
        rows.append((i, d, trend_ok, pullback))
    if not rows:
        print("no mappable signals"); return
    idx = np.array([r[0] for r in rows]); dirs = np.array([r[1] for r in rows])
    trend = np.array([r[2] for r in rows]); pb = np.array([r[3] for r in rows])
    print(f"{strategy} EURUSD signals mapped: {len(idx)} "
          f"({(dirs>0).sum()} BUY) | span {df['start_time'].values[idx.min()]} "
          f".. {df['start_time'].values[idx.max()]} | cost={COST}")

    cells = {
        'ALL': np.ones(len(idx), bool),
        'trend_ok': trend, 'trend_bad': ~trend,
        'pullback': pb, 'no_pullback': ~pb,
        'A-aligned (trend+pb)': trend & pb,
        'trend_ok, no_pb': trend & ~pb,
        'AGAINST (no trend+no pb)': ~trend & ~pb,
    }
    specs = [ExitSpec('brkt 15/30', 'bracket', sl=15, tp=30),
             ExitSpec('brkt 25/50', 'bracket', sl=25, tp=50),
             ExitSpec('sl25+t288', 'sl_time', sl=25, time_bars=288)]
    hdr = (f"{'cell':26s} {'n':>4s} | H144 exp %pos | H288 exp %pos" +
           "".join(f" | {s.name}: pf exp" for s in specs))
    print(hdr); print("-" * len(hdr))
    for nm, m in cells.items():
        if m.sum() < 8:
            print(f"{nm:26s} {m.sum():4d}  (too few)"); continue
        i_s, d_s = idx[m], dirs[m]
        line = f"{nm:26s} {m.sum():4d}"
        for H in (144, 288):
            r = fwd_net(df, i_s, d_s, H)
            line += f" | {r.mean():6.2f} {100*(r>0).mean():4.1f}"
        for s in specs:
            net, _ = sim_exit(df, i_s, d_s, s)
            st = stats(net)
            line += f" | {st['pf']:5.2f} {st['exp']:6.2f}"
        print(line, flush=True)

    # half-split robustness for the key cells
    med = np.median(idx)
    print("\nhalf-split (by signal date) — brkt 15/30 pf/exp:")
    for nm in ('ALL', 'trend_ok', 'trend_bad', 'trend_ok, no_pb', 'pullback'):
        m = cells[nm]
        for half, hm in (('H1', m & (idx <= med)), ('H2', m & (idx > med))):
            if hm.sum() < 8:
                print(f"  {nm:22s} {half}: n={hm.sum()} too few"); continue
            net, _ = sim_exit(df, idx[hm], dirs[hm], specs[0])
            st = stats(net)
            print(f"  {nm:22s} {half}: n={st['n']:4d} pf={st['pf']:5.2f} exp={st['exp']:6.2f}")


# ---------------------------------------------------------------- validate
def run_validate():
    """Per-year PF for the standalone winner; half-split for the overlay cell."""
    df = prep()
    rsi = df['rsi'].values
    up_t = df['h1_ema21'].values > df['h1_ema50'].values
    dn_t = df['h1_ema21'].values < df['h1_ema50'].values
    a_up = up_t & (rsi < 35); a_dn = dn_t & (rsi > 65)
    for spec in (ExitSpec('time 288b', 'time', time_bars=288),
                 ExitSpec('sl25 + time 288b', 'sl_time', sl=25, time_bars=288),
                 ExitSpec('bracket 30/60', 'bracket', sl=30, tp=60)):
        net, ts, dirs, bars = sim_sequential(df, a_up, a_dn, spec)
        years = pd.DatetimeIndex(ts).year
        print(f"\nA rsi5<35 1h-trend x {spec.name} — per year:")
        for y in sorted(set(years)):
            st = stats(net[years == y])
            print(f"  {y}: n={st['n']:4d} pf={st['pf']:5.2f} exp={st['exp']:6.2f}")


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['exits', 'entries', 'refine', 'combo',
                                    'combo2', 'overlay', 'validate'])
    ap.add_argument('--cost', type=float, default=2.0)
    ap.add_argument('--strategy', default='SMC_SIMPLE')
    args = ap.parse_args()
    COST = args.cost
    if args.cmd == 'exits':
        run_exits()
    elif args.cmd == 'entries':
        run_entries()
    elif args.cmd == 'refine':
        run_refine()
    elif args.cmd == 'combo':
        run_combo()
    elif args.cmd == 'combo2':
        run_combo2()
    elif args.cmd == 'overlay':
        run_overlay(args.strategy)
    else:
        run_validate()
