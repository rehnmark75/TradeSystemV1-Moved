"""Character-cell tagging for the stock scanner edge-map router.

A "character cell" is a coarse, causal description of the market character a
signal fired in, on four axes:

    trend_state    : 'range' | 'mid' | 'trend'   (from ADX)
    vol_regime     : 'low'   | 'normal' | 'high' (from ATR %)
    liquidity_tier : 'thin'  | 'normal' | 'high' (from relative volume)
    market_regime  : copied as-of from market_context.market_regime

The router learns which (scanner x cell) combos have forward edge, then a later
stage routes only edge-positive cells into the tradable pool. THIS MODULE IS
PURE DATA -- it classifies, it does not gate.

Two entry points:

  classify_cell(adx_14, atr_percent, relative_volume, market_regime) -> dict
      Pure function. No DB, no I/O. Robust to None -> the corresponding axis is
      returned as None. Never raises on bad/missing input.

  tag_signal_row(conn, ticker, signal_date, ...) -> dict
      Pulls the as-of metrics for one signal from stock_screening_metrics and
      market_context (causal: calculation_date <= signal_date) and returns the
      cell dict. Robust to missing rows -> None fields, never crashes.

Thresholds live here as named constants and are the SINGLE SOURCE OF TRUTH.
Migration 043 documents the same bins; keep them in sync if you tune here.

DB column mapping note: stock_screening_metrics stores ADX in a column named
`adx` (14-period). The `adx_14` argument name in classify_cell is kept for
readability; tag_signal_row feeds metrics.adx into it.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Threshold constants (SINGLE SOURCE OF TRUTH for the cell bins)
# ---------------------------------------------------------------------------
# Trend (ADX, 14-period):  range < 20  ;  20 <= mid < 25  ;  trend >= 25
ADX_RANGE_MAX = 20.0   # adx < ADX_RANGE_MAX          -> 'range'
ADX_TREND_MIN = 25.0   # adx >= ADX_TREND_MIN         -> 'trend'  (else 'mid')

# Volatility (ATR % of price):  low < 2  ;  2 <= normal < 4  ;  high >= 4
ATR_LOW_MAX  = 2.0     # atr% < ATR_LOW_MAX           -> 'low'
ATR_HIGH_MIN = 4.0     # atr% >= ATR_HIGH_MIN         -> 'high'   (else 'normal')

# Liquidity (relative volume):  thin < 1  ;  1 <= normal < 3  ;  high >= 3
RVOL_THIN_MAX = 1.0    # rvol < RVOL_THIN_MAX         -> 'thin'
RVOL_HIGH_MIN = 3.0    # rvol >= RVOL_HIGH_MIN        -> 'high'   (else 'normal')


# ---------------------------------------------------------------------------
# Internal coercion helper
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> Optional[float]:
    """Best-effort float coercion. Returns None for None / non-numeric / NaN.

    Handles Decimal (psycopg2 numeric), str, int, float. Never raises.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NaN check without importing math for a hot path
    if f != f:  # noqa: PLR0124 - NaN is the only value not equal to itself
        return None
    return f


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------
def _trend_state(adx: Optional[float]) -> Optional[str]:
    if adx is None:
        return None
    if adx < ADX_RANGE_MAX:
        return "range"
    if adx >= ADX_TREND_MIN:
        return "trend"
    return "mid"


def _vol_regime(atr_percent: Optional[float]) -> Optional[str]:
    if atr_percent is None:
        return None
    if atr_percent < ATR_LOW_MAX:
        return "low"
    if atr_percent >= ATR_HIGH_MIN:
        return "high"
    return "normal"


def _liquidity_tier(relative_volume: Optional[float]) -> Optional[str]:
    if relative_volume is None:
        return None
    if relative_volume < RVOL_THIN_MAX:
        return "thin"
    if relative_volume >= RVOL_HIGH_MIN:
        return "high"
    return "normal"


def classify_cell(
    adx_14: Any = None,
    atr_percent: Any = None,
    relative_volume: Any = None,
    market_regime: Any = None,
) -> dict:
    """Classify a market character cell from raw metrics. Pure, never raises.

    Args:
        adx_14: ADX(14) value (stock_screening_metrics.adx). None -> trend_state None.
        atr_percent: ATR as % of price (stock_screening_metrics.atr_percent).
        relative_volume: current/avg volume ratio (relative_volume).
        market_regime: as-of market_context.market_regime string (passed through).

    Returns:
        {
            'trend_state':    'range'|'mid'|'trend'|None,
            'vol_regime':     'low'|'normal'|'high'|None,
            'liquidity_tier': 'thin'|'normal'|'high'|None,
            'market_regime':  <str>|None,
        }
    """
    adx = _to_float(adx_14)
    atrp = _to_float(atr_percent)
    rvol = _to_float(relative_volume)

    mr: Optional[str]
    if market_regime is None:
        mr = None
    else:
        mr = str(market_regime).strip() or None

    return {
        "trend_state": _trend_state(adx),
        "vol_regime": _vol_regime(atrp),
        "liquidity_tier": _liquidity_tier(rvol),
        "market_regime": mr,
    }


# ---------------------------------------------------------------------------
# As-of DB tagging helper
# ---------------------------------------------------------------------------
# Causal as-of metric lookup: the most recent screening row for the ticker whose
# calculation_date is on or before the signal date. This guarantees no future
# data leaks into the cell classification.
_METRICS_SQL = """
    SELECT adx, atr_percent, relative_volume
    FROM stock_screening_metrics
    WHERE ticker = %(ticker)s
      AND calculation_date <= %(signal_date)s
    ORDER BY calculation_date DESC
    LIMIT 1
"""

_MARKET_SQL = """
    SELECT market_regime
    FROM market_context
    WHERE calculation_date <= %(signal_date)s
    ORDER BY calculation_date DESC
    LIMIT 1
"""

_EMPTY_CELL = {
    "trend_state": None,
    "vol_regime": None,
    "liquidity_tier": None,
    "market_regime": None,
}


def tag_signal_row(
    conn,
    ticker: str,
    signal_date,
    *,
    max_metric_staleness_days: Optional[int] = None,
) -> dict:
    """Compute the character cell for one signal, as-of its signal date.

    Causal: only uses stock_screening_metrics / market_context rows with
    calculation_date <= signal_date. Robust to missing metrics -- returns a cell
    dict with None fields rather than raising.

    Args:
        conn: an open psycopg2 connection (any cursor_factory; read as tuples).
        ticker: signal ticker, e.g. 'AAPL'.
        signal_date: date (or date-like) of the signal (stock_scanner_signals.signal_date).
        max_metric_staleness_days: if set, and the most-recent screening row is
            older than this many days before signal_date, the technical axes are
            treated as missing (None). market_regime is looked up independently
            and is not subject to this guard. None (default) = no staleness cap.

    Returns:
        Same dict shape as classify_cell().
    """
    if ticker is None or signal_date is None:
        return dict(_EMPTY_CELL)

    adx = atrp = rvol = None
    market_regime = None

    try:
        # Use a plain cursor and read by position so this works regardless of the
        # connection's configured cursor_factory (RealDictCursor etc.).
        with conn.cursor() as cur:
            cur.execute(_METRICS_SQL, {"ticker": ticker, "signal_date": signal_date})
            row = cur.fetchone()
            if row is not None:
                # row may be a tuple or a dict-like depending on cursor_factory
                if isinstance(row, dict):
                    adx = row.get("adx")
                    atrp = row.get("atr_percent")
                    rvol = row.get("relative_volume")
                else:
                    adx, atrp, rvol = row[0], row[1], row[2]

            cur.execute(_MARKET_SQL, {"signal_date": signal_date})
            mrow = cur.fetchone()
            if mrow is not None:
                market_regime = mrow["market_regime"] if isinstance(mrow, dict) else mrow[0]
    except Exception:
        # Never let a metric lookup crash the caller (emission path must be safe).
        # A failed lookup yields an all-None cell, same as "metrics missing".
        try:
            conn.rollback()
        except Exception:
            pass
        return dict(_EMPTY_CELL)

    # Optional staleness guard on the technical axes only.
    if max_metric_staleness_days is not None and (adx is not None or atrp is not None or rvol is not None):
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MAX(calculation_date)
                    FROM stock_screening_metrics
                    WHERE ticker = %(ticker)s AND calculation_date <= %(signal_date)s
                    """,
                    {"ticker": ticker, "signal_date": signal_date},
                )
                r = cur.fetchone()
                latest = (r[0] if not isinstance(r, dict) else list(r.values())[0]) if r else None
            if latest is not None:
                # signal_date and latest are date objects
                staleness = (signal_date - latest).days if hasattr(signal_date, "__sub__") else None
                if staleness is not None and staleness > max_metric_staleness_days:
                    adx = atrp = rvol = None
        except Exception:
            pass  # staleness check is best-effort; keep the metrics we have

    return classify_cell(
        adx_14=adx,
        atr_percent=atrp,
        relative_volume=rvol,
        market_regime=market_regime,
    )
