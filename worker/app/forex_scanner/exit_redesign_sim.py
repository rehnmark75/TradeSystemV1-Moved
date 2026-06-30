#!/usr/bin/env python3
"""
EXPERIMENTAL (Jun 30 2026) — Exit-redesign simulation harness.

Tests replacing fixed SL/TP exits on EXISTING strategy entry signals with:
  - policy 'fixed'    : today's model (fixed SL/TP bracket)  -> CONTROL
  - policy 'flip'     : run until an opposite-direction signal (always-in-market-ish)
  - policy 'adaptive' : run until opposite signal OR a rule-based scenario close

The entry-signal stream is supplied externally (alert_history or backtest-generated),
so this module is signal-source-agnostic. It maintains ONE persistent position and
re-evaluates it every bar — which the production backtest engine cannot do today.

This is the exit core that graduates into the live engine if policy C wins OOS.

NOT wired into any production path. Run via:
  docker exec -it task-worker python /app/forex_scanner/exit_redesign_sim.py ...
"""
from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

# ----------------------------------------------------------------------------
# Pip sizes and realistic per-pair round-trip costs (pips). Costs from
# reference_ig_realized_spread_jun15: ~2 EURUSD, ~4-5 GBPUSD/JPY crosses.
# ----------------------------------------------------------------------------
PIP_SIZE = {
    'CS.D.EURUSD.CEEM.IP': 0.0001, 'CS.D.GBPUSD.MINI.IP': 0.0001,
    'CS.D.AUDUSD.MINI.IP': 0.0001, 'CS.D.NZDUSD.MINI.IP': 0.0001,
    'CS.D.USDCAD.MINI.IP': 0.0001, 'CS.D.USDCHF.MINI.IP': 0.0001,
    'CS.D.EURGBP.MINI.IP': 0.0001,
    'CS.D.USDJPY.MINI.IP': 0.01, 'CS.D.EURJPY.MINI.IP': 0.01,
    'CS.D.AUDJPY.MINI.IP': 0.01, 'CS.D.GBPJPY.MINI.IP': 0.01,
    'CS.D.CFEGOLD.DUKAS.IP': 0.1, 'CS.D.CFEGOLD.CEE.IP': 0.1,
}
COST_PIPS = {  # round-trip, applied once per closed trade
    'CS.D.EURUSD.CEEM.IP': 2.0, 'CS.D.GBPUSD.MINI.IP': 2.5,
    'CS.D.AUDUSD.MINI.IP': 2.0, 'CS.D.NZDUSD.MINI.IP': 3.0,
    'CS.D.USDCAD.MINI.IP': 2.5, 'CS.D.USDCHF.MINI.IP': 2.5,
    'CS.D.EURGBP.MINI.IP': 2.0,
    'CS.D.USDJPY.MINI.IP': 2.0, 'CS.D.EURJPY.MINI.IP': 3.0,
    'CS.D.AUDJPY.MINI.IP': 3.0, 'CS.D.GBPJPY.MINI.IP': 4.0,
    'CS.D.CFEGOLD.DUKAS.IP': 35.0, 'CS.D.CFEGOLD.CEE.IP': 35.0,
}


def pip(epic: str) -> float:
    return PIP_SIZE.get(epic, 0.0001)


def cost(epic: str) -> float:
    return COST_PIPS.get(epic, 2.5)


# ----------------------------------------------------------------------------
# Indicators (numpy, Wilder-style ADX / ATR + EMA)
# ----------------------------------------------------------------------------
def _wilder(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    out[period - 1] = np.nanmean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def compute_indicators(df: pd.DataFrame, adx_period: int = 14,
                       atr_period: int = 14, ema_len: int = 50) -> pd.DataFrame:
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = _wilder(tr, atr_period)

    up = h - np.roll(h, 1); up[0] = 0
    dn = np.roll(l, 1) - l; dn[0] = 0
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_for_di = _wilder(tr, adx_period)
    plus_di = 100 * _wilder(plus_dm, adx_period) / np.where(atr_for_di == 0, np.nan, atr_for_di)
    minus_di = 100 * _wilder(minus_dm, adx_period) / np.where(atr_for_di == 0, np.nan, atr_for_di)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, plus_di + minus_di)
    adx = _wilder(np.nan_to_num(dx), adx_period)

    ema = pd.Series(c).ewm(span=ema_len, adjust=False).mean().values
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().values

    # RSI(14) Wilder
    delta = np.diff(c, prepend=c[0])
    gain = _wilder(np.where(delta > 0, delta, 0.0), 14)
    loss = _wilder(np.where(delta < 0, -delta, 0.0), 14)
    rs = gain / np.where(loss == 0, np.nan, loss)
    rsi = 100 - 100 / (1 + rs)

    df = df.copy()
    df['atr'] = atr
    df['adx'] = adx
    df['ema'] = ema
    df['ema200'] = ema200
    df['rsi'] = rsi
    return df


# ----------------------------------------------------------------------------
# Trade record + simulator
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    entry_idx: int
    entry_ts: pd.Timestamp
    direction: str          # 'BUY' / 'SELL'
    entry_px: float
    exit_idx: int = -1
    exit_ts: Optional[pd.Timestamp] = None
    exit_px: float = np.nan
    reason: str = ''
    pips_gross: float = 0.0
    pips_net: float = 0.0


@dataclass
class ExitParams:
    # fixed bracket (control)
    sl_pips: float = 20.0
    tp_pips: float = 40.0
    # adaptive close knobs
    adx_floor: float = 18.0          # close when ADX falls below this (trend gone)
    use_ema_confirm: bool = True     # require price on wrong side of EMA to close
    atr_trail_mult: float = 0.0      # >0 enables ATR give-back trail from peak (0=off)
    # safety disaster stop for flip/adaptive (pips, 0 = none)
    disaster_stop_pips: float = 0.0
    reverse_on_opposite: bool = True  # flip True = reverse; False = just go flat


def _dir_sign(direction: str) -> int:
    return 1 if direction == 'BUY' else -1


def simulate(signals: List[Tuple[pd.Timestamp, str]], df: pd.DataFrame,
             policy: str, epic: str, params: ExitParams) -> List[Trade]:
    """Single-position bar-by-bar state machine. signals: sorted (ts, dir)."""
    ps = pip(epic)
    ts_arr = df['start_time'].values
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    adx = df['adx'].values
    ema = df['ema'].values

    # map each signal to first bar index with start_time >= signal ts
    sig_dir = [None] * len(df)
    j = 0
    sigs = sorted(signals, key=lambda x: x[0])
    for sig_ts, sig_d in sigs:
        sig_ts64 = np.datetime64(sig_ts)
        while j < len(df) and ts_arr[j] < sig_ts64:
            j += 1
        if j >= len(df):
            break
        if sig_dir[j] is None:   # keep first signal landing on a bar
            sig_dir[j] = sig_d

    trades: List[Trade] = []
    pos: Optional[Trade] = None
    peak_pips = 0.0

    for i in range(len(df)):
        d = sig_dir[i]

        if pos is None:
            if d is not None:
                pos = Trade(entry_idx=i, entry_ts=pd.Timestamp(ts_arr[i]),
                            direction=d, entry_px=c[i])
                peak_pips = 0.0
            continue

        sign = _dir_sign(pos.direction)

        # ---- FIXED bracket: intrabar SL/TP on raw high/low ----
        if policy == 'fixed':
            if pos.direction == 'BUY':
                sl_px = pos.entry_px - params.sl_pips * ps
                tp_px = pos.entry_px + params.tp_pips * ps
                hit_sl = l[i] <= sl_px
                hit_tp = h[i] >= tp_px
            else:
                sl_px = pos.entry_px + params.sl_pips * ps
                tp_px = pos.entry_px - params.tp_pips * ps
                hit_sl = h[i] >= sl_px
                hit_tp = l[i] <= tp_px
            if hit_sl or hit_tp:
                # conservative: SL first if both in same bar
                exit_px = sl_px if hit_sl else tp_px
                reason = 'SL' if hit_sl else 'TP'
                _close(pos, i, ts_arr, exit_px, reason, sign, ps, epic)
                trades.append(pos)
                pos = None
            continue

        # ---- FLIP / ADAPTIVE: mark-to-market, evaluate exit rules ----
        cur_pips = sign * (c[i] - pos.entry_px) / ps
        peak_pips = max(peak_pips, cur_pips)

        opposite = d is not None and d != pos.direction

        # disaster stop (safety)
        disaster = False
        if params.disaster_stop_pips > 0:
            if pos.direction == 'BUY':
                disaster = l[i] <= pos.entry_px - params.disaster_stop_pips * ps
            else:
                disaster = h[i] >= pos.entry_px + params.disaster_stop_pips * ps

        adaptive_close = False
        if policy == 'adaptive':
            trend_gone = (not np.isnan(adx[i])) and adx[i] < params.adx_floor
            if params.use_ema_confirm:
                wrong_side = (c[i] < ema[i]) if pos.direction == 'BUY' else (c[i] > ema[i])
                trend_gone = trend_gone and wrong_side
            trail_hit = False
            if params.atr_trail_mult > 0 and not np.isnan(atr_at(df, i)):
                give_back = params.atr_trail_mult * atr_at(df, i) / ps
                trail_hit = peak_pips > 0 and cur_pips <= peak_pips - give_back
            adaptive_close = trend_gone or trail_hit

        if disaster:
            _close(pos, i, ts_arr, pos.entry_px - sign * params.disaster_stop_pips * ps,
                   'DISASTER', sign, ps, epic)
            trades.append(pos); pos = None
            if opposite and params.reverse_on_opposite:
                pos = Trade(i, pd.Timestamp(ts_arr[i]), d, c[i]); peak_pips = 0.0
            continue

        if opposite:
            _close(pos, i, ts_arr, c[i], 'OPPOSITE', sign, ps, epic)
            trades.append(pos); pos = None
            if params.reverse_on_opposite:
                pos = Trade(i, pd.Timestamp(ts_arr[i]), d, c[i]); peak_pips = 0.0
            continue

        if adaptive_close:
            _close(pos, i, ts_arr, c[i], 'ADAPTIVE', sign, ps, epic)
            trades.append(pos); pos = None
            continue

    # close any open position at last bar
    if pos is not None:
        i = len(df) - 1
        _close(pos, i, ts_arr, c[i], 'EOD', _dir_sign(pos.direction), ps, epic)
        trades.append(pos)
    return trades


def atr_at(df: pd.DataFrame, i: int) -> float:
    return df['atr'].values[i]


def _close(t: Trade, i: int, ts_arr, exit_px: float, reason: str,
           sign: int, ps: float, epic: str):
    t.exit_idx = i
    t.exit_ts = pd.Timestamp(ts_arr[i])
    t.exit_px = exit_px
    t.reason = reason
    t.pips_gross = sign * (exit_px - t.entry_px) / ps
    t.pips_net = t.pips_gross - cost(epic)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return dict(n=0, wr=0, pf=0, total=0, avg_win=0, avg_loss=0)
    pips = np.array([t.pips_net for t in trades])
    wins = pips[pips > 0]; losses = pips[pips <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')
    return dict(
        n=len(trades),
        wr=round(100 * len(wins) / len(trades), 1),
        pf=round(pf, 2),
        total=round(pips.sum(), 1),
        avg_win=round(wins.mean(), 1) if len(wins) else 0,
        avg_loss=round(losses.mean(), 1) if len(losses) else 0,
    )


def split_metrics(trades: List[Trade], split_ts: pd.Timestamp) -> Tuple[Dict, Dict]:
    is_t = [t for t in trades if t.entry_ts < split_ts]
    oos_t = [t for t in trades if t.entry_ts >= split_ts]
    return metrics(is_t), metrics(oos_t)


# ----------------------------------------------------------------------------
# Candle loader
# ----------------------------------------------------------------------------
def load_candles(epic: str, timeframe: int, start: str, end: str) -> pd.DataFrame:
    import config
    from forex_scanner.core.database import DatabaseManager
    dbm = DatabaseManager(config.DATABASE_URL)
    q = """
        SELECT start_time, open, high, low, close
        FROM ig_candles_backtest
        WHERE epic = %(epic)s AND timeframe = %(tf)s
          AND start_time >= %(start)s AND start_time < %(end)s
        ORDER BY start_time
    """
    df = pd.read_sql(q, dbm.get_engine(),
                     params={'epic': epic, 'tf': timeframe, 'start': start, 'end': end})
    for col in ('open', 'high', 'low', 'close'):
        df[col] = df[col].astype(float)
    df['start_time'] = pd.to_datetime(df['start_time'])
    return df.reset_index(drop=True)


def load_alert_signals(epic: str, strategy: str) -> List[Tuple[pd.Timestamp, str]]:
    """Load the real emitted entry-signal stream from alert_history (forex DB).
    Normalizes BULL/BEAR -> BUY/SELL. Returns sorted (ts, dir)."""
    import config
    from forex_scanner.core.database import DatabaseManager
    dbm = DatabaseManager(config.DATABASE_URL)
    q = """
        SELECT alert_timestamp, signal_type
        FROM alert_history
        WHERE epic = %(epic)s AND strategy = %(strat)s
        ORDER BY alert_timestamp
    """
    df = pd.read_sql(q, dbm.get_engine(), params={'epic': epic, 'strat': strategy})
    out = []
    for ts, st in zip(df['alert_timestamp'], df['signal_type']):
        s = str(st).upper()
        d = 'BUY' if s in ('BULL', 'BUY', 'LONG') else 'SELL' if s in ('BEAR', 'SELL', 'SHORT') else None
        if d:
            out.append((pd.Timestamp(ts), d))
    return out


PAIRS_8 = [
    'CS.D.EURUSD.CEEM.IP', 'CS.D.USDJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP',
    'CS.D.EURJPY.MINI.IP', 'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP',
    'CS.D.GBPUSD.MINI.IP', 'CS.D.NZDUSD.MINI.IP',
]

_DBM = None
_CANDLE_CACHE: Dict[str, pd.DataFrame] = {}


def _dbm():
    global _DBM
    if _DBM is None:
        import config
        from forex_scanner.core.database import DatabaseManager
        _DBM = DatabaseManager(config.DATABASE_URL)
    return _DBM


def candles_cached(epic: str, timeframe: int) -> Optional[pd.DataFrame]:
    """Load a wide 1h window per epic ONCE, with indicators; slice per use."""
    key = f"{epic}:{timeframe}"
    if key not in _CANDLE_CACHE:
        df = pd.read_sql(
            """SELECT start_time, open, high, low, close FROM ig_candles_backtest
               WHERE epic=%(e)s AND timeframe=%(tf)s
                 AND start_time >= '2025-06-01' AND start_time < '2026-07-01'
               ORDER BY start_time""",
            _dbm().get_engine(), params={'e': epic, 'tf': timeframe})
        if len(df) < 200:
            _CANDLE_CACHE[key] = None
        else:
            for c in ('open', 'high', 'low', 'close'):
                df[c] = df[c].astype(float)
            df['start_time'] = pd.to_datetime(df['start_time'])
            _CANDLE_CACHE[key] = compute_indicators(df.reset_index(drop=True))
    return _CANDLE_CACHE[key]


def discover_epics(strategy: str, min_sig: int) -> List[Tuple[str, int]]:
    df = pd.read_sql(
        """SELECT epic, COUNT(*) n FROM alert_history
           WHERE strategy=%(s)s AND epic IS NOT NULL
           GROUP BY epic HAVING COUNT(*) >= %(m)s ORDER BY n DESC""",
        _dbm().get_engine(), params={'s': strategy, 'm': min_sig})
    return list(zip(df['epic'], df['n']))


def gate_signals(sigs, df, adx_lo=0, adx_hi=999, hr_lo=0, hr_hi=23,
                 rsi_lo=0, rsi_hi=100):
    """Keep only signals whose entry-bar context is in the skill zone."""
    ts = df['start_time'].values.astype('datetime64[ns]')
    adx = df['adx'].values; rsi = df['rsi'].values
    n = len(df)
    out = []
    for sig_ts, d in sigs:
        i = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(sig_ts))))
        if i >= n:
            continue
        if not (adx_lo <= adx[i] <= adx_hi):
            continue
        if not (rsi_lo <= rsi[i] <= rsi_hi):
            continue
        if not (hr_lo <= pd.Timestamp(sig_ts).hour <= hr_hi):
            continue
        out.append((sig_ts, d))
    return out


def run_strategy(strategy: str, timeframe: int, params: ExitParams,
                 min_sig: int = 40, gate: dict = None) -> Dict[str, list]:
    """Pool all of a strategy's epics; return {policy: [trades]}."""
    agg = {p: [] for p in ('fixed', 'flip', 'adaptive')}
    for epic, _ in discover_epics(strategy, min_sig):
        if epic not in PIP_SIZE:
            continue
        sigs = load_alert_signals(epic, strategy)
        if len(sigs) < min_sig:
            continue
        df = candles_cached(epic, timeframe)
        if df is None:
            continue
        if gate:
            sigs = gate_signals(sigs, df, **gate)
        for pol in agg:
            agg[pol].extend(simulate(sigs, df, pol, epic, params))
    return agg


def run_all(strategies: List[str], timeframe: int, params: ExitParams, min_sig: int,
            gate: dict = None, split_frac: float = 0.6):
    tag = f"  GATE={gate}" if gate else "  (no gate)"
    print(f"\n{'='*100}\nEXIT-REDESIGN BREADTH TEST — {len(strategies)} strategies, "
          f"alert_history streams, tf={timeframe}m, per-pair cost{tag}\n{'='*100}")
    print(f"{'strategy':30s} {'fixed n/pf':>14s} {'flip n/pf':>14s} {'adaptive n/pf':>16s}  winner")
    print('-' * 100)
    grand = {p: [] for p in ('fixed', 'flip', 'adaptive')}
    wins = {p: 0 for p in ('fixed', 'flip', 'adaptive')}
    rows = []
    for strat in strategies:
        agg = run_strategy(strat, timeframe, params, min_sig, gate=gate)
        m = {p: metrics(agg[p]) for p in agg}
        if m['fixed']['n'] == 0:
            continue
        for p in grand:
            grand[p].extend(agg[p])
        # winner by PF among policies with >=20 trades
        cand = {p: m[p]['pf'] for p in m if m[p]['n'] >= 20}
        win = max(cand, key=cand.get) if cand else 'fixed'
        wins[win] += 1
        rows.append((strat, m, win))
        print(f"{strat:30s} {m['fixed']['n']:5d}/{m['fixed']['pf']:5.2f}  "
              f"{m['flip']['n']:5d}/{m['flip']['pf']:5.2f}  "
              f"{m['adaptive']['n']:6d}/{m['adaptive']['pf']:5.2f}    {win}")
    print('-' * 100)
    all_ts = sorted(t.entry_ts for p in grand for t in grand[p])
    split_ts = all_ts[int(len(all_ts) * split_frac)] if all_ts else None
    print(f"GRAND POOLED (every signal, every strategy)   IS/OOS split @ {split_ts.date() if split_ts is not None else '-'}:")
    for p in ('fixed', 'flip', 'adaptive'):
        g = metrics(grand[p])
        mi, mo = split_metrics(grand[p], split_ts) if split_ts is not None else ({}, {})
        print(f"  {p:10s} ALL n={g['n']:5d}/pf={g['pf']:5.2f}/pips={g['total']:8.0f}   "
              f"IS pf={mi.get('pf',0):5.2f}   OOS n={mo.get('n',0):4d}/pf={mo.get('pf',0):5.2f}/pips={mo.get('total',0):7.0f}")
    print(f"\nPer-strategy winner counts (PF, n>=20): {wins}")
    print(f"Strategies tested: {len(rows)}")


def run_phase1a(strategy: str, timeframe: int, split_frac: float, params: ExitParams):
    """Alert_history stream -> compare fixed/flip/adaptive per pair + portfolio."""
    print(f"\n{'='*88}\nPHASE 1a — exit-redesign on REAL {strategy} alert_history signals "
          f"(tf={timeframe}m, cost=per-pair, split={split_frac:.0%})\n{'='*88}")
    agg = {p: [] for p in ('fixed', 'flip', 'adaptive')}
    for epic in PAIRS_8:
        sigs = load_alert_signals(epic, strategy)
        if len(sigs) < 30:
            print(f"\n{epic:24s}  SKIP (only {len(sigs)} signals)")
            continue
        start = (sigs[0][0] - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
        end = (sigs[-1][0] + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        df = load_candles(epic, timeframe, start, end)
        if len(df) < 200:
            print(f"\n{epic:24s}  SKIP (only {len(df)} candles)")
            continue
        df = compute_indicators(df)
        split_ts = sigs[int(len(sigs) * split_frac)][0]
        print(f"\n{epic:24s}  n_sig={len(sigs)}  candles={len(df)}  split@{split_ts.date()}")
        print(f"  {'policy':10s} {'ALL n/wr/pf/pips':>26s}   {'IS pf':>8s}  {'OOS n/pf/pips':>18s}")
        for pol in ('fixed', 'flip', 'adaptive'):
            trades = simulate(sigs, df, pol, epic, params)
            m = metrics(trades)
            mi, mo = split_metrics(trades, split_ts)
            agg[pol].extend(trades)
            print(f"  {pol:10s} {m['n']:4d}/{m['wr']:5.1f}/{m['pf']:5.2f}/{m['total']:8.0f}"
                  f"   {mi['pf']:8.2f}  {mo['n']:3d}/{mo['pf']:5.2f}/{mo['total']:7.0f}")
    print(f"\n{'-'*88}\nPORTFOLIO (all pairs pooled):")
    for pol in ('fixed', 'flip', 'adaptive'):
        m = metrics(agg[pol])
        print(f"  {pol:10s}  n={m['n']:5d}  wr={m['wr']:5.1f}  pf={m['pf']:5.2f}  "
              f"pips={m['total']:9.0f}  avgW={m['avg_win']:6.1f}  avgL={m['avg_loss']:6.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true',
                    help='Run a synthetic sanity check of the three policies')
    ap.add_argument('--phase1a', action='store_true', help='Run alert_history comparison')
    ap.add_argument('--all', action='store_true', help='Breadth test across all strategies')
    ap.add_argument('--min-sig', type=int, default=40)
    ap.add_argument('--gate', action='store_true', help='Apply stacked context gate (adx18-28+hr0-16+rsi50-70)')
    ap.add_argument('--gate-adx', action='store_true', help='Apply ADX 18-28 gate only')
    ap.add_argument('--strategy', default='SMC_SIMPLE')
    ap.add_argument('--timeframe', type=int, default=60)
    ap.add_argument('--split', type=float, default=0.7)
    ap.add_argument('--adx-floor', type=float, default=18.0)
    ap.add_argument('--sl', type=float, default=15.0)
    ap.add_argument('--tp', type=float, default=25.0)
    ap.add_argument('--atr-trail', type=float, default=0.0)
    ap.add_argument('--disaster', type=float, default=0.0)
    args = ap.parse_args()
    if args.phase1a:
        p = ExitParams(sl_pips=args.sl, tp_pips=args.tp, adx_floor=args.adx_floor,
                       atr_trail_mult=args.atr_trail, disaster_stop_pips=args.disaster)
        run_phase1a(args.strategy, args.timeframe, args.split, p)
        sys.exit(0)
    if args.all:
        p = ExitParams(sl_pips=args.sl, tp_pips=args.tp, adx_floor=args.adx_floor,
                       atr_trail_mult=args.atr_trail, disaster_stop_pips=args.disaster)
        gate = None
        if args.gate:
            gate = dict(adx_lo=18, adx_hi=28, hr_lo=0, hr_hi=16, rsi_lo=50, rsi_hi=70)
        elif args.gate_adx:
            gate = dict(adx_lo=18, adx_hi=28)
        sdf = pd.read_sql(
            """SELECT strategy FROM alert_history WHERE strategy IS NOT NULL
               GROUP BY strategy HAVING COUNT(*) >= %(m)s ORDER BY COUNT(*) DESC""",
            _dbm().get_engine(), params={'m': args.min_sig})
        run_all(list(sdf['strategy']), args.timeframe, p, args.min_sig, gate=gate)
        sys.exit(0)
    if args.selftest:
        # synthetic uptrend then downtrend, with alternating signals
        n = 500
        t = pd.date_range('2024-01-01', periods=n, freq='1h')
        price = np.concatenate([np.linspace(1.10, 1.15, n // 2),
                                np.linspace(1.15, 1.10, n - n // 2)])
        df = pd.DataFrame({'start_time': t, 'open': price, 'high': price + 0.0005,
                           'low': price - 0.0005, 'close': price})
        df = compute_indicators(df)
        sigs = [(t[10], 'BUY'), (t[260], 'SELL')]
        for pol in ('fixed', 'flip', 'adaptive'):
            tr = simulate(sigs, df, pol, 'CS.D.EURUSD.CEEM.IP', ExitParams())
            print(pol, metrics(tr))
