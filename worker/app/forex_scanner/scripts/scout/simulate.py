"""Minimal trade simulator for the strategy scout.

Deliberately NOT reusing `core/trading/vsl_trailing_simulator.py`: that module
layers spread modeling, VSL staging, and live-path coupling that biases the
comparison. The scout wants identical fixtures across all templates so PF
differences trace to signal logic alone.

Contract
--------
Input: OHLC dataframe + int signals array of same length (+1 BUY / -1 SELL / 0 flat).
Output: list of trade dicts with entry/exit prices, outcome, pips, bars held,
regime label at entry (stamped externally by the orchestrator).

Semantics
---------
- Entry at signal bar's close (conservative vs next-bar open; matches live
  scanner which evaluates at candle close then places market order).
- Walk forward bar-by-bar. First SL or TP hit wins. Tie (same bar hits
  both) -> SL (conservative).
- Cooldown: `cooldown_bars` bars must elapse after exit before a new
  signal is re-armed.
- Max hold: `max_hold_bars` bars -> TIMEOUT, exit at close of the
  max-hold bar. Default 288 = 1 day on 5m.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def simulate(
    df: pd.DataFrame,
    signals: np.ndarray,
    sl_pips: float,
    tp_pips: float,
    pip_value: float,
    cooldown_bars: int = 6,
    max_hold_bars: int = 288,
) -> List[Dict]:
    """Walk the frame, emit trades.

    Args:
        df: OHLC frame with at least 'open','high','low','close' columns,
            DatetimeIndex. Contiguous candles expected (caller handles gaps).
        signals: int array, len(df). +1 = BUY, -1 = SELL, 0 = flat.
        sl_pips: stop-loss distance (positive pips).
        tp_pips: take-profit distance (positive pips).
        pip_value: 0.01 for JPY pairs, 0.0001 for most others (gold 0.1).
        cooldown_bars: bars after exit before re-arming.
        max_hold_bars: force exit if neither SL nor TP hit by this many bars.

    Returns:
        List of dicts — one per trade — with keys:
            entry_idx, exit_idx, entry_ts, exit_ts, direction,
            entry, exit, sl_price, tp_price, outcome ('TP'|'SL'|'TIMEOUT'),
            pips, bars_held.
    """
    if df is None or len(df) == 0:
        return []
    if len(signals) != len(df):
        raise ValueError(f"signals length {len(signals)} != df length {len(df)}")

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index

    trades: List[Dict] = []
    i = 0
    n = len(df)
    cooldown_until = -1

    while i < n:
        if i <= cooldown_until:
            i += 1
            continue
        s = signals[i]
        if s == 0:
            i += 1
            continue

        direction = "BUY" if s == 1 else "SELL"
        entry_price = float(closes[i])
        if direction == "BUY":
            sl_price = entry_price - sl_pips * pip_value
            tp_price = entry_price + tp_pips * pip_value
        else:
            sl_price = entry_price + sl_pips * pip_value
            tp_price = entry_price - tp_pips * pip_value

        # Walk forward
        outcome = "TIMEOUT"
        exit_idx = min(i + max_hold_bars, n - 1)
        exit_price = float(closes[exit_idx])
        for j in range(i + 1, min(i + 1 + max_hold_bars, n)):
            hi = float(highs[j])
            lo = float(lows[j])
            if direction == "BUY":
                hit_sl = lo <= sl_price
                hit_tp = hi >= tp_price
            else:
                hit_sl = hi >= sl_price
                hit_tp = lo <= tp_price

            if hit_sl and hit_tp:
                # Tie: SL wins (conservative)
                outcome = "SL"
                exit_idx = j
                exit_price = sl_price
                break
            if hit_sl:
                outcome = "SL"
                exit_idx = j
                exit_price = sl_price
                break
            if hit_tp:
                outcome = "TP"
                exit_idx = j
                exit_price = tp_price
                break

        sign = 1 if direction == "BUY" else -1
        pips = (exit_price - entry_price) * sign / pip_value

        trades.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "entry_ts": idx[i],
            "exit_ts": idx[exit_idx],
            "direction": direction,
            "entry": entry_price,
            "exit": exit_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "outcome": outcome,
            "pips": pips,
            "bars_held": exit_idx - i,
        })

        cooldown_until = exit_idx + cooldown_bars
        i = exit_idx + 1

    return trades


def aggregate(trades: List[Dict]) -> Dict:
    """Roll up a trades list into PF/WR/avg win/avg loss / Sharpe / max_dd.

    Sharpe is approximated as mean(pips) / std(pips) with no annualization —
    purely for intra-scout ranking, not a true Sharpe.

    Max drawdown is computed on cumulative pips."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0.0, "pf": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "total_pips": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0,
            "sl_count": 0, "tp_count": 0, "timeout_count": 0,
        }

    df = pd.DataFrame(trades)
    wins = df[df["pips"] > 0]
    losses = df[df["pips"] < 0]

    win_pnl = float(wins["pips"].sum()) if not wins.empty else 0.0
    loss_pnl = float(-losses["pips"].sum()) if not losses.empty else 0.0
    pf = (win_pnl / loss_pnl) if loss_pnl > 0 else float("inf") if win_pnl > 0 else 0.0

    cum = df["pips"].cumsum()
    peak = cum.cummax()
    drawdown = (peak - cum).max()

    returns = df["pips"].values
    sharpe = float(np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0.0

    return {
        "n_trades": len(df),
        "win_rate": float((df["pips"] > 0).mean()),
        "pf": float(pf) if np.isfinite(pf) else 999.0,
        "avg_win": float(wins["pips"].mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses["pips"].mean()) if not losses.empty else 0.0,
        "total_pips": float(df["pips"].sum()),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown),
        "sl_count": int((df["outcome"] == "SL").sum()),
        "tp_count": int((df["outcome"] == "TP").sum()),
        "timeout_count": int((df["outcome"] == "TIMEOUT").sum()),
    }


def bootstrap_pf_ci(
    trades: List[Dict],
    n_bootstrap: int = 1000,
    block_size: int = 5,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Block bootstrap CI on PF. Block size > 1 respects serial correlation
    between consecutive trades (e.g., a cluster of losses during a range
    break). Returns lower, upper, median."""
    if not trades or len(trades) < 2 * block_size:
        return {"lower": 0.0, "upper": 0.0, "median": 0.0}

    df = pd.DataFrame(trades)
    pips = df["pips"].values
    n = len(pips)
    n_blocks = n // block_size
    if n_blocks < 2:
        return {"lower": 0.0, "upper": 0.0, "median": 0.0}

    rng = np.random.default_rng(42)
    pfs = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Pick n_blocks block start indices with replacement
        starts = rng.integers(0, n - block_size + 1, n_blocks)
        sample = np.concatenate([pips[s:s + block_size] for s in starts])
        wins_sum = sample[sample > 0].sum()
        loss_sum = -sample[sample < 0].sum()
        if loss_sum > 0:
            pfs[b] = wins_sum / loss_sum
        elif wins_sum > 0:
            pfs[b] = 10.0  # cap infinities
        else:
            pfs[b] = 0.0

    alpha = 1 - confidence
    lower = float(np.percentile(pfs, 100 * alpha / 2))
    upper = float(np.percentile(pfs, 100 * (1 - alpha / 2)))
    median = float(np.percentile(pfs, 50))
    return {"lower": lower, "upper": upper, "median": median}
