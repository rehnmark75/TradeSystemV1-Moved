#!/usr/bin/env python3
"""
EXPERIMENTAL (Jul 9 2026) — FreedomScalp V3 (Gaussian) Pine-port lab for GOLD.

Faithful Python port of the TradingView Pine v6 strategy
"FreedomScalp V3 - Gaussian (Price Units)":
  - N-pole Gaussian filter on close (Ehlers-style recursive), alpha from length
  - linreg(len, offset) flatten of the Gaussian output
  - SuperTrend(factor=0.15, atrPeriod=21) computed on the SMOOTHED line
    (ATR from raw chart H/L/C), trend = smoothed > supertrend
  - entry on trend flip at bar close -> filled NEXT bar open (Pine default)
  - TP/SL are RAW PRICE distances anchored at the SIGNAL bar close (as in the
    Pine: strategy.exit limit/stop are set from `close` when the order is placed)
  - optional TradingView-style trailing stop (activation + offset, price units)
  - filters: EMA regime gate, spike filter, session window, ADX gate, ATR
    expansion gate, MACD/QQE confirmations

Methodology (house rules):
  - de-bias: entry sets benchmarked against time/session-matched RANDOM entries
    with the SAME exit model (an exit profitable on random entries = artifact)
  - signal at bar close, entry next bar open; no lookahead
  - all open positions force-closed at end of each split window
  - intrabar ambiguity resolved PESSIMISTICALLY: stop checked before TP,
    trailing watermark updated AFTER exit checks
  - costs: round-trip price units subtracted per trade (default 0.40 = IG-ish
    gold spread ~0.3 + slippage). Sensitivity via --cost.

Data:
  ig_candles_backtest CS.D.CFEGOLD.DUKAS.IP 5m  2020-01 .. 2025-09  (Dukascopy)
  ig_candles_backtest CS.D.CFEGOLD.CEE.IP  5m  2026-01 .. 2026-06  (live IG)

Splits (Dukas): IS 2020-2022 | OOS-A 2023-2024 | OOS-B 2025-01..2025-09
Extra live-source check: LIVE-26 = CEE.IP 2026-01..2026-06

Subcommands:
  baseline   TV-default settings across all splits + random nulls
  ablate     one-at-a-time filter toggles on the default core (per split)
  sweep      core Gaussian param grid, selected on IS, validated OOS
  sltp       TP/SL/trailing grid (incl. ATR-scaled variant) for a chosen core
  final      single named config, full per-split report + random null

Run:
  docker exec task-worker python /app/forex_scanner/freedomscalp_lab.py baseline
"""
from __future__ import annotations
import sys, argparse, itertools, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

EPIC_DUKAS = 'CS.D.CFEGOLD.DUKAS.IP'
EPIC_LIVE = 'CS.D.CFEGOLD.CEE.IP'
COST = 0.40         # round-trip, price units (USD). override --cost
SEED = 11
N_RANDOM = 12       # random-entry replicates for the null

SPLITS = [
    # (name, epic, start, end)
    ('IS',      EPIC_DUKAS, '2020-01-01', '2022-12-31'),
    ('OOS-A',   EPIC_DUKAS, '2023-01-01', '2024-12-31'),
    ('OOS-B',   EPIC_DUKAS, '2025-01-01', '2025-09-17'),
    ('LIVE-26', EPIC_LIVE,  '2026-01-18', '2026-06-24'),
]

# ---------------------------------------------------------------- data
def _engine():
    from core.database import DatabaseManager
    import config
    return DatabaseManager(config.DATABASE_URL).get_engine()


_CACHE: dict = {}


def load(epic: str, tf: int = 5) -> pd.DataFrame:
    key = (epic, tf)
    if key in _CACHE:
        return _CACHE[key]
    df = pd.read_sql(
        """SELECT start_time, open, high, low, close FROM ig_candles_backtest
           WHERE epic=%(e)s AND timeframe=%(tf)s ORDER BY start_time""",
        _engine(), params={'e': epic, 'tf': tf})
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.drop_duplicates('start_time').reset_index(drop=True)
    for c in ('open', 'high', 'low', 'close'):
        df[c] = df[c].astype(float)
    _CACHE[key] = df
    return df


# ---------------------------------------------------------------- pine ports
def rma(arr: np.ndarray, period: int) -> np.ndarray:
    """Pine ta.rma (Wilder)."""
    out = np.full(len(arr), np.nan)
    a = np.nan_to_num(arr, nan=0.0)
    if len(arr) < period:
        return out
    out[period - 1] = a[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(arr)):
        out[i] = out[i - 1] + alpha * (a[i] - out[i - 1])
    return out


def atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return rma(tr, period)


def gaussian_alpha(length: int, order: int) -> float:
    freq = 2.0 * math.pi / length
    b = (1.0 - math.cos(freq)) / (1.414 ** (2.0 / order) - 1.0)
    return -b + math.sqrt(b * b + 2.0 * b)


def gaussian_smooth(src: np.ndarray, poles: int, a: float) -> np.ndarray:
    """N-pole recursive Gaussian, exact port of the Pine gaussianSmooth()."""
    n = len(src)
    v = np.zeros(n)
    oma = 1.0 - a
    if poles == 1:
        for i in range(n):
            v[i] = a * src[i] + oma * (v[i - 1] if i >= 1 else 0.0)
    elif poles == 2:
        for i in range(n):
            v[i] = (a * a * src[i]
                    + 2 * oma * (v[i - 1] if i >= 1 else 0.0)
                    - oma * oma * (v[i - 2] if i >= 2 else 0.0))
    elif poles == 3:
        for i in range(n):
            v[i] = (a ** 3 * src[i]
                    + 3 * oma * (v[i - 1] if i >= 1 else 0.0)
                    - 3 * oma ** 2 * (v[i - 2] if i >= 2 else 0.0)
                    + oma ** 3 * (v[i - 3] if i >= 3 else 0.0))
    elif poles == 4:
        for i in range(n):
            v[i] = (a ** 4 * src[i]
                    + 4 * oma * (v[i - 1] if i >= 1 else 0.0)
                    - 6 * oma ** 2 * (v[i - 2] if i >= 2 else 0.0)
                    + 4 * oma ** 3 * (v[i - 3] if i >= 3 else 0.0)
                    - oma ** 4 * (v[i - 4] if i >= 4 else 0.0))
    else:
        raise ValueError('poles 1..4')
    return v


def linreg(src: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Pine ta.linreg: LSRL over window, evaluated at x = length-1-offset
    (x=0 oldest bar .. x=length-1 newest). Vectorized via convolution."""
    n = len(src)
    x = np.arange(length, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    # window w[k] = src[i-length+1+k]; slope = sum((x-xm)*w)/denom
    # value at xe = length-1-offset: intercept + slope*xe = wmean + slope*(xe-xm)
    xe = (length - 1 - offset) - xm
    coef = (1.0 / length) + (x - xm) * xe / denom          # per-window weights
    out = np.full(n, np.nan)
    if n >= length:
        # convolve: out[i] = sum_k coef[k]*src[i-length+1+k]
        conv = np.convolve(src, coef[::-1], mode='valid')
        out[length - 1:] = conv
    return out


def pine_supertrend(src: np.ndarray, atr_arr: np.ndarray, factor: float):
    """Port of the script's pine_supertrend(src, factor, atrPeriod); atr_arr
    precomputed from chart H/L/C. Returns st array."""
    n = len(src)
    st = np.full(n, np.nan)
    up_f = np.zeros(n); dn_f = np.zeros(n)   # final (ratcheted) bands
    for i in range(n):
        av = atr_arr[i]
        if np.isnan(av):
            av = 0.0
        up = src[i] + factor * av
        dn = src[i] - factor * av
        pdn = dn_f[i - 1] if i >= 1 else 0.0
        pup = up_f[i - 1] if i >= 1 else 0.0
        psrc = src[i - 1] if i >= 1 else src[i]
        dn = dn if (dn > pdn or psrc < pdn) else pdn
        up = up if (up < pup or psrc > pup) else pup
        dn_f[i] = dn; up_f[i] = up
        pst = st[i - 1] if i >= 1 else np.nan
        prev_atr_na = (i == 0) or np.isnan(atr_arr[i - 1])
        if prev_atr_na:
            d = 1
        elif pst == pup:
            d = -1 if src[i] > up else 1
        else:
            d = 1 if src[i] < dn else -1
        st[i] = dn if d == -1 else up
    return st


# ---------------------------------------------------------------- indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['close'].values
    df['atr14'] = atr(df, 14)
    df['atr21'] = atr(df, 21)
    cs = pd.Series(c)
    df['ema50'] = cs.ewm(span=50, adjust=False).mean().values
    df['ema200'] = cs.ewm(span=200, adjust=False).mean().values
    # ADX(14) wilder
    h, l = df['high'].values, df['low'].values
    up = h - np.roll(h, 1); up[0] = 0
    dn = np.roll(l, 1) - l; dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr_di = rma(tr, 14)
    with np.errstate(invalid='ignore', divide='ignore'):
        pdi = 100 * rma(pdm, 14) / atr_di
        mdi = 100 * rma(mdm, 14) / atr_di
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    df['adx'] = rma(np.nan_to_num(dx), 14)
    # MACD hist 12/26/9
    ema12 = cs.ewm(span=12, adjust=False).mean()
    ema26 = cs.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()).values
    # QQE smoothed RSI: ema(rsi(6),5)
    delta = np.diff(c, prepend=c[0])
    gain = rma(np.where(delta > 0, delta, 0.0), 6)
    loss = rma(np.where(delta < 0, -delta, 0.0), 6)
    with np.errstate(invalid='ignore', divide='ignore'):
        rsi6 = 100 - 100 / (1 + gain / loss)
    df['qqe_rsi'] = pd.Series(rsi6).ewm(span=5, adjust=False).mean().values
    # ATR expansion gate inputs
    df['atr14_sma50'] = pd.Series(df['atr14'].values).rolling(50).mean().values
    # session hour in UTC (db timestamps are UTC; Pine default 0600-2030 UTC+2)
    df['hhmm'] = df['start_time'].dt.hour * 100 + df['start_time'].dt.minute
    return df


SESSIONS = {                       # UTC windows (Pine defaults were UTC+2)
    'default': (400, 1830),        # 0600-2030 UTC+2
    'overlap': (1200, 1500),       # 1400-1700 UTC+2
    'london':  (800, 1100),
    'ny':      (1330, 1700),
    'off':     (0, 2400),
}


def session_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    a, b = SESSIONS[name]
    return (df['hhmm'].values >= a) & (df['hhmm'].values < b)


# ---------------------------------------------------------------- signals
def compute_signals(df: pd.DataFrame, period: int, poles: int, ll: int, lo: int,
                    ema_filter=True, spike=True, spike_mult=3.0,
                    adx_gate=None, atr_exp=None, macd=False, qqe=False,
                    min_confs=0, session='default', mid_trend=False):
    c = df['close'].values
    a = gaussian_alpha(period, poles)
    g = gaussian_smooth(c, poles, a)
    final = linreg(g, ll, lo)
    fin = np.nan_to_num(final, nan=c[0])
    st = pine_supertrend(fin, df['atr21'].values, 0.15)
    trend = np.where(fin > st, 1, -1)
    ptrend = np.roll(trend, 1); ptrend[0] = trend[0]
    if mid_trend:
        pfin = np.roll(fin, 1); pfin[0] = fin[0]
        ranging = (np.where(fin > pfin, 1, -1) * trend) < 0
        pranging = np.roll(ranging, 1); pranging[0] = False
        gbuy = (trend > 0) & (~ranging) & pranging
        gsell = (trend < 0) & (~ranging) & pranging
    else:
        gbuy = (trend > 0) & (ptrend < 0)
        gsell = (trend < 0) & (ptrend > 0)

    buy_score = ((macd & (df['macd_hist'].values > 0)).astype(int)
                 + (qqe & (df['qqe_rsi'].values - 50 > 3.0)).astype(int))
    sell_score = ((macd & (df['macd_hist'].values < 0)).astype(int)
                  + (qqe & (df['qqe_rsi'].values - 50 < -3.0)).astype(int))

    ema50 = df['ema50'].values
    ebuy = (c > ema50) if ema_filter else np.ones(len(c), bool)
    esell = (c < ema50) if ema_filter else np.ones(len(c), bool)
    patr = np.roll(df['atr14'].values, 1); patr[0] = np.nan
    is_spike = (df['high'].values - df['low'].values) > patr * spike_mult
    sp_ok = ~np.nan_to_num(is_spike, nan=False) if spike else np.ones(len(c), bool)
    reg_ok = np.ones(len(c), bool)
    if adx_gate is not None:
        reg_ok &= df['adx'].values > adx_gate
    if atr_exp is not None:
        reg_ok &= df['atr14'].values > df['atr14_sma50'].values * atr_exp
    sess = session_mask(df, session)
    warm = np.arange(len(c)) > max(200, ll + period * poles)

    long_sig = gbuy & (buy_score >= min_confs) & ebuy & sp_ok & reg_ok & sess & warm
    short_sig = gsell & (sell_score >= min_confs) & esell & sp_ok & reg_ok & sess & warm
    return long_sig, short_sig, sess


# ---------------------------------------------------------------- simulator
def _tv_walk_bar(side, o_, h_, l_, c_, tp_lvl, sl_lvl, use_ts, act_lvl, off,
                 state):
    """TradingView broker-emulator path for one bar: O->L->H->C if close>=open
    else O->H->L->C, price monotonic within each leg, trailing stop ratchets
    continuously along the path. Mutates state dict {peak, trail_on, trail}.
    Returns (exit_px, reason) or (None, None)."""
    legs = ([(o_, l_), (l_, h_), (h_, c_)] if c_ >= o_
            else [(o_, h_), (h_, l_), (l_, c_)])
    for p1, p2 in legs:
        lo_, hi_ = (p1, p2) if p1 <= p2 else (p2, p1)
        favorable = (p2 >= p1) if side == 1 else (p2 <= p1)
        if favorable:
            # favorable leg: TP / trail-activation-and-update as price advances
            if side == 1:
                if hi_ >= tp_lvl:
                    return tp_lvl, 'tp'
                if use_ts:
                    state['peak'] = max(state['peak'], hi_)
                    if not state['trail_on'] and state['peak'] >= act_lvl:
                        state['trail_on'] = True
                    if state['trail_on']:
                        state['trail'] = max(state['trail'], state['peak'] - off)
            else:
                if lo_ <= tp_lvl:
                    return tp_lvl, 'tp'
                if use_ts:
                    state['peak'] = min(state['peak'], lo_)
                    if not state['trail_on'] and state['peak'] <= act_lvl:
                        state['trail_on'] = True
                    if state['trail_on']:
                        state['trail'] = min(state['trail'], state['peak'] + off)
        else:
            # adverse leg: hard SL / trailing stop
            if side == 1:
                stop = max(sl_lvl, state['trail']) if (use_ts and state['trail_on']) else sl_lvl
                if lo_ <= stop:
                    return min(p1, stop), 'sl'
            else:
                stop = min(sl_lvl, state['trail']) if (use_ts and state['trail_on']) else sl_lvl
                if hi_ >= stop:
                    return max(p1, stop), 'sl'
    return None, None


def simulate(df: pd.DataFrame, long_sig, short_sig, sess_mask,
             tp: float, sl: float, use_ts=True, ts_act=1.0, ts_off=0.5,
             cost=COST, session_close=True, atr_scale=None, fills='pessimistic'):
    """Single-position sim, Pine-faithful:
    signal at bar i close -> entry at open[i+1]; TP/SL anchored at close[i].
    atr_scale: if set, tp/sl/ts_* are multiples of atr14[i] instead of $.
    fills: 'pessimistic' (stop-first, trail updated after checks) or
           'tv' (TradingView broker-emulator OHLC path with intrabar trail)."""
    o, h, l, c = (df[k].values for k in ('open', 'high', 'low', 'close'))
    atr14 = df['atr14'].values
    n = len(df)
    trades = []
    i = 0
    while i < n - 1:
        if not (long_sig[i] or short_sig[i]):
            i += 1
            continue
        side = 1 if long_sig[i] else -1
        scale = atr14[i] if atr_scale else 1.0
        if atr_scale and (np.isnan(scale) or scale <= 0):
            i += 1
            continue
        _tp, _sl = tp * scale, sl * scale
        _act, _off = ts_act * scale, ts_off * scale
        anchor = c[i]
        entry = o[i + 1]
        tp_lvl = anchor + side * _tp
        sl_lvl = anchor - side * _sl
        peak = entry
        trail_on = False
        trail_lvl = -np.inf if side == 1 else np.inf
        exit_px = None; exit_j = None; reason = None
        if fills == 'tv':
            state = {'peak': entry, 'trail_on': False,
                     'trail': -np.inf if side == 1 else np.inf}
            act_lvl = entry + side * _act
            j = i + 1
            while j < n:
                px, why = _tv_walk_bar(side, o[j], h[j], l[j], c[j],
                                       tp_lvl, sl_lvl, use_ts, act_lvl,
                                       _off, state)
                if px is not None:
                    exit_px, reason, exit_j = px, why, j
                    break
                if session_close and not sess_mask[j]:
                    exit_px = c[j]; reason = 'session'; exit_j = j
                    break
                j += 1
            if exit_px is None:
                exit_px = c[n - 1]; reason = 'end'; exit_j = n - 1
            pnl = side * (exit_px - entry) - cost
            trades.append((i, exit_j, side, entry, exit_px, pnl, reason))
            i = exit_j + 1
            continue
        j = i + 1
        while j < n:
            stop_lvl = sl_lvl
            if use_ts and trail_on:
                stop_lvl = max(sl_lvl, trail_lvl) if side == 1 else min(sl_lvl, trail_lvl)
            if side == 1:
                if l[j] <= stop_lvl:                     # pessimistic: stop first
                    exit_px = min(o[j], stop_lvl) if o[j] < stop_lvl else stop_lvl
                    reason = 'sl'; exit_j = j; break
                if h[j] >= tp_lvl:
                    exit_px = max(o[j], tp_lvl) if o[j] > tp_lvl else tp_lvl
                    reason = 'tp'; exit_j = j; break
                if use_ts:
                    peak = max(peak, h[j])
                    if not trail_on and peak >= entry + _act:
                        trail_on = True
                    if trail_on:
                        trail_lvl = max(trail_lvl, peak - _off)
            else:
                if h[j] >= stop_lvl:
                    exit_px = max(o[j], stop_lvl) if o[j] > stop_lvl else stop_lvl
                    reason = 'sl'; exit_j = j; break
                if l[j] <= tp_lvl:
                    exit_px = min(o[j], tp_lvl) if o[j] < tp_lvl else tp_lvl
                    reason = 'tp'; exit_j = j; break
                if use_ts:
                    peak = min(peak, l[j])
                    if not trail_on and peak <= entry - _act:
                        trail_on = True
                    if trail_on:
                        trail_lvl = min(trail_lvl, peak + _off)
            if session_close and not sess_mask[j]:
                exit_px = c[j]; reason = 'session'; exit_j = j; break
            j += 1
        if exit_px is None:                              # force close at end
            exit_px = c[n - 1]; reason = 'end'; exit_j = n - 1
        pnl = side * (exit_px - entry) - cost
        trades.append((i, exit_j, side, entry, exit_px, pnl, reason))
        i = exit_j + 1                                   # single position
    return trades


def stats(trades, label=''):
    if not trades:
        return {'label': label, 'n': 0, 'pf': np.nan, 'wr': np.nan,
                'exp': np.nan, 'net': 0.0, 'dd': np.nan}
    pnl = np.array([t[5] for t in trades])
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    eq = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {'label': label, 'n': len(pnl),
            'pf': gp / gl if gl > 0 else np.inf,
            'wr': 100.0 * (pnl > 0).mean(),
            'exp': float(pnl.mean()), 'net': float(pnl.sum()), 'dd': dd}


def fmt(s):
    return (f"{s['label']:<26} n={s['n']:>5}  PF={s['pf']:.2f}  WR={s['wr']:5.1f}%"
            f"  exp=${s['exp']:+.3f}  net=${s['net']:+8.1f}  maxDD=${s['dd']:.1f}")


# ---------------------------------------------------------------- null model
def random_null(df, n_entries, sess, sim_kwargs, seed=SEED, reps=N_RANDOM):
    """Time-matched random entries (same count, session-eligible bars),
    same exit model. Returns mean PF / expectancy across reps."""
    rng = np.random.default_rng(seed)
    eligible = np.where(sess & (np.arange(len(df)) > 250))[0]
    eligible = eligible[eligible < len(df) - 2]
    pfs, exps = [], []
    for _ in range(reps):
        idx = np.sort(rng.choice(eligible, size=min(n_entries, len(eligible)),
                                 replace=False))
        side = rng.integers(0, 2, size=len(idx))
        ls = np.zeros(len(df), bool); ss = np.zeros(len(df), bool)
        ls[idx[side == 1]] = True; ss[idx[side == 0]] = True
        tr = simulate(df, ls, ss, sess, **sim_kwargs)
        s = stats(tr)
        if s['n'] > 0:
            pfs.append(min(s['pf'], 5.0)); exps.append(s['exp'])
    return float(np.mean(pfs)), float(np.mean(exps))


# ---------------------------------------------------------------- runners
def split_frames(tf: int = 5):
    out = []
    for name, epic, a, b in SPLITS:
        df = load(epic, tf)
        m = (df['start_time'] >= a) & (df['start_time'] <= b + ' 23:59')
        d = df.loc[m].reset_index(drop=True)
        if len(d) > 1000:
            out.append((name, add_indicators(d)))
    return out


DEFAULT_CORE = dict(period=8, poles=2, ll=10, lo=3)
DEFAULT_EXIT = dict(tp=4.2, sl=6.0, use_ts=True, ts_act=1.0, ts_off=0.5)


def run_config(dfs, core, sig_kwargs, exit_kwargs, cost, with_null=False, tag=''):
    rows = []
    for name, df in dfs:
        ls, ss, sess = compute_signals(df, **core, **sig_kwargs)
        tr = simulate(df, ls, ss, sess, cost=cost, **exit_kwargs)
        s = stats(tr, f'{tag}{name}')
        if with_null and s['n'] >= 5:
            npf, nexp = random_null(df, s['n'], sess,
                                    dict(cost=cost, **exit_kwargs))
            s['null_pf'] = npf; s['null_exp'] = nexp
        rows.append(s)
    return rows


def cmd_baseline(args):
    dfs = split_frames(args.tf)
    ex = dict(DEFAULT_EXIT, fills=args.fills)
    print(f"=== BASELINE: TV defaults (P8/2p/LL10/LO3, EMA50+spike+session, "
          f"TP4.2/SL6.0, trail 1.0/0.5), cost=${args.cost} RT, fills={args.fills} ===")
    rows = run_config(dfs, DEFAULT_CORE, dict(), ex, args.cost,
                      with_null=True)
    for s in rows:
        line = fmt(s)
        if 'null_pf' in s:
            line += f"   | random-null PF={s['null_pf']:.2f} exp=${s['null_exp']:+.3f}"
        print(line)
    print("\n--- no-trailing variant ---")
    ek = dict(ex); ek['use_ts'] = False
    for s in run_config(dfs, DEFAULT_CORE, dict(), ek, args.cost):
        print(fmt(s))
    print("\n--- zero-cost sanity (is there any gross edge?) ---")
    for s in run_config(dfs, DEFAULT_CORE, dict(), ex, 0.0):
        print(fmt(s))


def simulate_hf(df5, long_sig, short_sig, df1, tp, sl, use_ts=True,
                ts_act=1.0, ts_off=0.5, cost=COST, fills='tv',
                session='default'):
    """Signals on 5m, exit path walked on 1m bars — arbitrates the intrabar
    fill-model dispute (at 1m the TV path assumption barely matters)."""
    ts5 = df5['start_time'].values
    ts1 = df1['start_time'].values
    o1, h1, l1, c1 = (df1[k].values for k in ('open', 'high', 'low', 'close'))
    hhmm1 = df1['start_time'].dt.hour.values * 100 + df1['start_time'].dt.minute.values
    a, b = SESSIONS[session]
    in_sess1 = (hhmm1 >= a) & (hhmm1 < b)
    c5 = df5['close'].values
    n5, n1 = len(df5), len(df1)
    sig_idx = np.where(long_sig | short_sig)[0]
    trades = []
    busy_until = -1
    for i in sig_idx:
        if i >= n5 - 1:
            break
        t_entry = ts5[i] + np.timedelta64(5, 'm')
        k0 = np.searchsorted(ts1, t_entry)
        if k0 >= n1:
            break
        if k0 <= busy_until:
            continue
        side = 1 if long_sig[i] else -1
        anchor = c5[i]
        entry = o1[k0]
        tp_lvl = anchor + side * tp
        sl_lvl = anchor - side * sl
        state = {'peak': entry, 'trail_on': False,
                 'trail': -np.inf if side == 1 else np.inf}
        act_lvl = entry + side * ts_act
        exit_px = None; reason = None; kx = None
        for k in range(k0, n1):
            if fills == 'tv':
                px, why = _tv_walk_bar(side, o1[k], h1[k], l1[k], c1[k],
                                       tp_lvl, sl_lvl, use_ts, act_lvl,
                                       ts_off, state)
            else:
                px, why = None, None
                stop = sl_lvl
                if use_ts and state['trail_on']:
                    stop = max(sl_lvl, state['trail']) if side == 1 else min(sl_lvl, state['trail'])
                if side == 1 and l1[k] <= stop:
                    px, why = min(o1[k], stop) if o1[k] < stop else stop, 'sl'
                elif side == -1 and h1[k] >= stop:
                    px, why = max(o1[k], stop) if o1[k] > stop else stop, 'sl'
                elif side == 1 and h1[k] >= tp_lvl:
                    px, why = tp_lvl, 'tp'
                elif side == -1 and l1[k] <= tp_lvl:
                    px, why = tp_lvl, 'tp'
                elif use_ts:
                    if side == 1:
                        state['peak'] = max(state['peak'], h1[k])
                        if not state['trail_on'] and state['peak'] >= act_lvl:
                            state['trail_on'] = True
                        if state['trail_on']:
                            state['trail'] = max(state['trail'], state['peak'] - ts_off)
                    else:
                        state['peak'] = min(state['peak'], l1[k])
                        if not state['trail_on'] and state['peak'] <= act_lvl:
                            state['trail_on'] = True
                        if state['trail_on']:
                            state['trail'] = min(state['trail'], state['peak'] + ts_off)
            if px is not None:
                exit_px, reason, kx = px, why, k
                break
            if not in_sess1[k]:
                exit_px, reason, kx = c1[k], 'session', k
                break
        if exit_px is None:
            exit_px, reason, kx = c1[n1 - 1], 'end', n1 - 1
        pnl = side * (exit_px - entry) - cost
        trades.append((i, kx, side, entry, exit_px, pnl, reason))
        busy_until = kx
    return trades


def cmd_m1(args):
    """5m signals + 1m exit walk on the 2026 live-IG window, both fill models,
    vs the pure-5m results."""
    eng = _engine()
    d5 = load(EPIC_LIVE, 5)
    m = (d5['start_time'] >= '2026-01-18') & (d5['start_time'] <= '2026-06-24 23:59')
    d5 = add_indicators(d5.loc[m].reset_index(drop=True))
    d1 = pd.read_sql(
        """SELECT start_time, open, high, low, close FROM ig_candles
           WHERE epic=%(e)s AND timeframe=1
             AND start_time BETWEEN '2026-01-18' AND '2026-06-24 23:59'
           ORDER BY start_time""",
        eng, params={'e': EPIC_LIVE})
    d1['start_time'] = pd.to_datetime(d1['start_time'])
    d1 = d1.drop_duplicates('start_time').reset_index(drop=True)
    for cc in ('open', 'high', 'low', 'close'):
        d1[cc] = d1[cc].astype(float)
    print(f"5m bars: {len(d5)}, 1m bars: {len(d1)}")
    ls, ss, sess = compute_signals(d5, **DEFAULT_CORE)
    print(f"=== LIVE-26 window, TV-default config, cost=${args.cost} ===")
    for fills in ('tv', 'pessimistic'):
        tr5 = simulate(d5, ls, ss, sess, cost=args.cost,
                       **dict(DEFAULT_EXIT, fills=fills))
        print(fmt(stats(tr5, f'5m-exits  fills={fills}')))
    for fills in ('tv', 'pessimistic'):
        tr1 = simulate_hf(d5, ls, ss, d1, cost=args.cost, fills=fills,
                          **DEFAULT_EXIT)
        print(fmt(stats(tr1, f'1m-exits  fills={fills}')))


def cmd_years(args):
    """Per-calendar-year stats — window-sensitivity check (TV only loads the
    last few months of 5m data, so compare recent slices vs full history)."""
    ex = dict(DEFAULT_EXIT, fills=args.fills)
    years = [(str(y), EPIC_DUKAS, f'{y}-01-01', f'{y}-12-31')
             for y in range(2020, 2026)]
    years.append(('2026(IG)', EPIC_LIVE, '2026-01-18', '2026-06-24'))
    # plus TV-sized recent windows on the freshest data
    years.append(('last3m-IG', EPIC_LIVE, '2026-03-24', '2026-06-24'))
    years.append(('2025H1', EPIC_DUKAS, '2025-01-01', '2025-06-30'))
    years.append(('2025Q3', EPIC_DUKAS, '2025-07-01', '2025-09-17'))
    print(f"=== PER-YEAR / RECENT-WINDOW, TV defaults, cost=${args.cost}, "
          f"fills={args.fills} ===")
    for name, epic, a, b in years:
        df = load(epic, args.tf)
        m = (df['start_time'] >= a) & (df['start_time'] <= b + ' 23:59')
        d = df.loc[m].reset_index(drop=True)
        if len(d) < 2000:
            continue
        d = add_indicators(d)
        ls, ss, sess = compute_signals(d, **DEFAULT_CORE)
        tr = simulate(d, ls, ss, sess, cost=args.cost, **ex)
        print(fmt(stats(tr, name)))


def cmd_ablate(args):
    dfs = split_frames()
    variants = [
        ('default', dict()),
        ('no-EMA-gate', dict(ema_filter=False)),
        ('no-spike', dict(spike=False)),
        ('session=off', dict(session='off')),
        ('session=overlap', dict(session='overlap')),
        ('session=london', dict(session='london')),
        ('session=ny', dict(session='ny')),
        ('+ADX>20', dict(adx_gate=20.0)),
        ('+ADX>25', dict(adx_gate=25.0)),
        ('+ATRexp>1.0', dict(atr_exp=1.0)),
        ('+ATRexp>1.2', dict(atr_exp=1.2)),
        ('+MACD conf', dict(macd=True, min_confs=1)),
        ('+QQE conf', dict(qqe=True, min_confs=1)),
        ('+MACD+QQE (2)', dict(macd=True, qqe=True, min_confs=2)),
        ('mid-trend sigs', dict(mid_trend=True)),
    ]
    print(f"=== ABLATION on TV-default core/exit, cost=${args.cost} ===")
    for vname, kw in variants:
        rows = run_config(dfs, DEFAULT_CORE, kw, DEFAULT_EXIT, args.cost,
                          tag=f'{vname:<18} ')
        for s in rows:
            print(fmt(s))
        print()


def cmd_sweep(args):
    dfs = split_frames()
    is_dfs = [x for x in dfs if x[0] == 'IS']
    oos_dfs = [x for x in dfs if x[0] != 'IS']
    grid = list(itertools.product([4, 6, 8, 10, 14, 20], [1, 2, 3],
                                  [6, 10, 14], [0, 3]))
    print(f"=== CORE SWEEP {len(grid)} configs on IS (2020-22), "
          f"cost=${args.cost}, TV-default filters/exit ===")
    results = []
    for period, poles, ll, lo in grid:
        core = dict(period=period, poles=poles, ll=ll, lo=lo)
        s = run_config(is_dfs, core, dict(), DEFAULT_EXIT, args.cost)[0]
        results.append((core, s))
    results.sort(key=lambda r: -(r[1]['exp'] * math.sqrt(max(r[1]['n'], 1))))
    print(f"{'period':>6} {'poles':>5} {'ll':>4} {'lo':>3}   IS-stats")
    for core, s in results[:12]:
        print(f"{core['period']:>6} {core['poles']:>5} {core['ll']:>4} "
              f"{core['lo']:>3}   {fmt(s)}")
    print("\n=== top-5 IS configs validated on OOS ===")
    for core, s in results[:5]:
        print(f"--- P{core['period']}/p{core['poles']}/LL{core['ll']}/LO{core['lo']} ---")
        for r in run_config(oos_dfs, core, dict(), DEFAULT_EXIT, args.cost):
            print(fmt(r))


def cmd_sltp(args):
    dfs = split_frames()
    core = dict(period=args.period, poles=args.poles, ll=args.ll, lo=args.lo)
    sigk = json.loads(args.sig) if args.sig else {}
    print(f"=== SL/TP GRID on core {core}, filters {sigk or 'TV-default'}, "
          f"cost=${args.cost} ===")
    fixed_grid = [
        dict(tp=4.2, sl=6.0), dict(tp=6.0, sl=6.0), dict(tp=8.0, sl=6.0),
        dict(tp=6.0, sl=4.0), dict(tp=10.0, sl=8.0), dict(tp=12.0, sl=6.0),
        dict(tp=15.0, sl=10.0), dict(tp=20.0, sl=10.0),
    ]
    for base in fixed_grid:
        for ts in (False, True):
            ek = dict(base, use_ts=ts, ts_act=max(1.0, base['tp'] * 0.25),
                      ts_off=max(0.5, base['tp'] * 0.12))
            tag = f"$tp{base['tp']}/sl{base['sl']}{'/tr' if ts else '   '} "
            for s in run_config(dfs, core, sigk, ek, args.cost, tag=tag):
                print(fmt(s))
        print()
    print("=== ATR-SCALED variants (multiples of ATR14 at signal) ===")
    atr_grid = [(1.0, 1.5), (1.5, 1.5), (2.0, 1.5), (2.0, 2.0), (3.0, 2.0),
                (4.0, 2.5)]
    for tpm, slm in atr_grid:
        for ts in (False, True):
            ek = dict(tp=tpm, sl=slm, use_ts=ts, ts_act=tpm * 0.3,
                      ts_off=tpm * 0.15, atr_scale=True)
            tag = f"ATR tp{tpm}/sl{slm}{'/tr' if ts else '   '} "
            for s in run_config(dfs, core, sigk, ek, args.cost, tag=tag):
                print(fmt(s))
        print()


def cmd_final(args):
    dfs = split_frames()
    core = dict(period=args.period, poles=args.poles, ll=args.ll, lo=args.lo)
    sigk = json.loads(args.sig) if args.sig else {}
    ek = json.loads(args.exit) if args.exit else dict(DEFAULT_EXIT)
    print(f"=== FINAL config core={core} sig={sigk} exit={ek} cost=${args.cost} ===")
    for s in run_config(dfs, core, sigk, ek, args.cost, with_null=True):
        line = fmt(s)
        if 'null_pf' in s:
            line += f"   | random-null PF={s['null_pf']:.2f} exp=${s['null_exp']:+.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['baseline', 'ablate', 'sweep', 'sltp',
                                    'final', 'years', 'm1'])
    ap.add_argument('--cost', type=float, default=COST)
    ap.add_argument('--tf', type=int, default=5)
    ap.add_argument('--fills', choices=['pessimistic', 'tv'],
                    default='pessimistic')
    ap.add_argument('--period', type=int, default=8)
    ap.add_argument('--poles', type=int, default=2)
    ap.add_argument('--ll', type=int, default=10)
    ap.add_argument('--lo', type=int, default=3)
    ap.add_argument('--sig', type=str, default='',
                    help='JSON signal kwargs, e.g. {"adx_gate":20,"session":"overlap"}')
    ap.add_argument('--exit', type=str, default='',
                    help='JSON exit kwargs, e.g. {"tp":2.0,"sl":1.5,"use_ts":false,"atr_scale":true}')
    args = ap.parse_args()
    {'baseline': cmd_baseline, 'ablate': cmd_ablate, 'sweep': cmd_sweep,
     'sltp': cmd_sltp, 'final': cmd_final, 'years': cmd_years,
     'm1': cmd_m1}[args.cmd](args)


if __name__ == '__main__':
    main()
