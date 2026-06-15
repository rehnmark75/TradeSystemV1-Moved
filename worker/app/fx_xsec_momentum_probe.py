#!/usr/bin/env python3
"""
FX Cross-Sectional Momentum Probe — pre-registered experiment
See: /home/hr/Projects/TradeSystemV1/FX_CARRY_XSECTIONAL_SCOPE.md

Tests G10 cross-sectional momentum: rank all 8 currencies (USD, EUR, GBP, AUD, NZD,
JPY, CAD, CHF) by trailing K-month USD-denominated return, long top-K, short bottom-K,
equal-weight, dollar-neutral, monthly + weekly rebalance.

Run inside task-worker:
    python /app/fx_xsec_momentum_probe.py [--smoke]

--smoke flag runs ONE config (K=3, top-2, monthly) over the full window for sanity
checking — does NOT run the full K×basket×freq grid.  You run the full job:

    python /app/fx_xsec_momentum_probe.py > /tmp/fx_xsec_results.txt 2>&1

Pre-registered decision bar (locked before looking at results):
    GO if OOS Sharpe >= 0.5 AND OOS max_dd <= 25% AND OOS total_return > 0
       for at least ONE clean config (K, basket_size, rebalance_freq).
    NO-GO otherwise.

    Context: cross-sectional momentum has a weaker prior than carry
    (scope doc: "hold OOS strictly"), so we use Sharpe >= 0.5 rather than
    a softer threshold. "max-DD tolerable" is defined as <= 25% of notional,
    which roughly corresponds to 2-3 years' worth of carry income lost in a
    bad unwind — a defensible stopping point for a G10 portfolio.
"""

import sys
import argparse
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Pre-registered decision constants (DO NOT change after first run)
# ---------------------------------------------------------------------------

OOS_SHARPE_MIN   = 0.5    # OOS Sharpe >= this
OOS_MAX_DD_MAX   = 0.25   # OOS max-drawdown <= 25% of dollar-notional (fraction)
OOS_TOTAL_RET_MIN = 0.0   # OOS total return > 0

# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------

# 7 IG/Dukascopy pair epics; USD is constant-1 numeraire (no epic needed)
PAIRS = [
    # (currency, epic, invert?)
    # invert=True  → USD-price = 1/pair_close  (USD-first pairs)
    # invert=False → USD-price = pair_close    (non-USD-first pairs)
    ("EUR", "CS.D.EURUSD.CEEM.IP", False),
    ("GBP", "CS.D.GBPUSD.MINI.IP", False),
    ("AUD", "CS.D.AUDUSD.MINI.IP", False),
    ("NZD", "CS.D.NZDUSD.MINI.IP", False),
    ("JPY", "CS.D.USDJPY.MINI.IP", True),
    ("CAD", "CS.D.USDCAD.MINI.IP", True),
    ("CHF", "CS.D.USDCHF.MINI.IP", True),
    # USD is the numeraire; no DB lookup needed
]

ALL_CURRENCIES = [ccy for ccy, _, _ in PAIRS] + ["USD"]

# Pip size (used only for cost conversion; cost applied per currency per rebalance)
# FX majors = 0.0001 per pip; JPY pairs = 0.01 per pip.
PIP_SIZE = {
    "EUR": 0.0001,   # EURUSD
    "GBP": 0.0001,   # GBPUSD
    "AUD": 0.0001,   # AUDUSD
    "NZD": 0.0001,   # NZDUSD
    "JPY": 0.01,     # USDJPY (inverted)
    "CAD": 0.0001,   # USDCAD (inverted; pip is still 0.0001 of USD price = 1/USDCAD)
    "CHF": 0.0001,   # USDCHF (inverted; same reasoning)
    "USD": 0.0001,   # never traded; placeholder
}

# Approximate NATIVE pair price for pip → fractional-return cost conversion.
#
# Formula: cost_fraction = COST_PIPS * PIP_SIZE[ccy] / NATIVE_PAIR_PRICE[ccy]
#
# For inverted pairs (JPY/CAD/CHF), the pip size is defined on the NATIVE quote
# (e.g. USDJPY pip = 0.01), so we divide by the NATIVE pair price (~150 for USDJPY),
# NOT by the inverted USD price (1/150 ≈ 0.0067).
# Using the inverted price would overcharge by factor ~(price)^2: for JPY that's
# ~150^2 = 22,500× too large (catastrophic).
#
# After inversion the cost translates correctly into USD-price space because
# the round-trip spread in the native quote (0.02 on USDJPY) equals
# the same round-trip in inverted space as a fraction of the inverted price.
# Numerically: 2pips / 150 ≈ 1.33e-4 (fractional cost ≈ 1.3bps). Correct.
PRICE_APPROX = {
    "EUR": 1.10,     # EURUSD native: ~1.10
    "GBP": 1.27,     # GBPUSD native: ~1.27
    "AUD": 0.65,     # AUDUSD native: ~0.65
    "NZD": 0.60,     # NZDUSD native: ~0.60
    "JPY": 150.0,    # USDJPY native: ~150  (pip 0.01 / 150 ≈ 1.3bps)
    "CAD": 1.36,     # USDCAD native: ~1.36 (pip 0.0001 / 1.36 ≈ 0.7bps)
    "CHF": 0.90,     # USDCHF native: ~0.90 (pip 0.0001 / 0.90 ≈ 1.1bps)
    "USD": 1.0,      # never traded; cost = 0 regardless
}

COST_PIPS = 2.0  # 2 pips per leg per turned-over position

WINDOW_START = "2020-01-01"
WINDOW_END   = "2026-06-12"

# IS / OOS split
IS_START  = "2020-01-01"
IS_END    = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END   = "2026-06-12"

# Grid to evaluate in full run
K_MONTHS        = [1, 3, 6, 12]
BASKET_SIZES    = [2, 3]          # top-k long, bottom-k short
REBALANCE_FREQS = ["monthly", "weekly"]

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

DB_HOST = "postgres"
DB_PORT = 5432
DB_NAME = "forex"
DB_USER = "postgres"
DB_PASSWORD = "postgres"


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )


def load_4h_candles(conn, epic: str) -> pd.DataFrame:
    """
    Load 4H candles from ig_candles_backtest for one epic.
    Mirrors htf_edge_probe.py's load_4h_candles pattern.
    """
    sql = """
        SELECT start_time, close
        FROM ig_candles_backtest
        WHERE epic = %s
          AND timeframe = 240
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
    """
    df = pd.read_sql(
        sql, conn,
        params=(epic, WINDOW_START, WINDOW_END),
        parse_dates=["start_time"],
    )
    df = df.set_index("start_time")
    return df


# ---------------------------------------------------------------------------
# Price series construction
# ---------------------------------------------------------------------------

def build_usd_price_series(
    pair_closes: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Build 8 USD-denominated price series at 4H frequency.

    For each currency:
      EUR/GBP/AUD/NZD  → price = close (USD per currency unit, no inversion)
      JPY/CAD/CHF      → price = 1/close (invert USD-first pair to get USD per CCY)
      USD              → price = 1.0 constant

    CRITICAL: invert via PRICE (1/close), never via return.
    """
    series_dict = {}

    for ccy, epic, invert in PAIRS:
        raw = pair_closes[ccy]
        if invert:
            price = 1.0 / raw
        else:
            price = raw.copy()
        series_dict[ccy] = price

    prices = pd.DataFrame(series_dict)

    # USD numeraire: constant 1.0 on EVERY row of the union index.
    # Assigning AFTER DataFrame construction ensures USD covers all timestamps
    # (not just EUR's timestamps), so resample("ME").last() never returns NaN for USD.
    prices["USD"] = 1.0

    return prices


def resample_to_month_end(prices_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4H price series to month-end closes.

    Uses last available 4H close within each calendar month.
    USD column (constant 1.0) survives the resample correctly.
    """
    monthly = prices_4h.resample("ME").last()
    return monthly


def resample_to_week_end(prices_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4H price series to week-end (Friday) closes.
    """
    weekly = prices_4h.resample("W-FRI").last()
    return weekly


# ---------------------------------------------------------------------------
# Signal and portfolio construction
# ---------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simple percentage return per period: r_t = price_t / price_{t-1} - 1.
    USD column has returns of 0.0 throughout (correct numeraire behaviour).
    """
    return prices.pct_change()


def compute_trailing_return(prices: pd.DataFrame, k_periods: int) -> pd.DataFrame:
    """
    Trailing K-period cumulative return (price relative), computed as:
        signal_t = prices_t / prices_{t-k} - 1

    This is the RAW signal that we rank at the END of period t to choose
    the portfolio held DURING period t+1.

    k_periods is already expressed in number of rebalance periods (months or
    weeks); caller is responsible for converting K months → weeks if needed.

    ANTI-LOOK-AHEAD: we do NOT shift here; the caller handles the t→t+1 lag.
    """
    return prices / prices.shift(k_periods) - 1


def assign_weights(
    signal_row: pd.Series,
    basket_size: int,
    n_currencies: int = 8,
) -> pd.Series:
    """
    Given a cross-section of signals (one value per currency, for one date),
    return equal-weight signed positions:
      +1/basket_size for the top-`basket_size` currencies
      -1/basket_size for the bottom-`basket_size` currencies
      0 for the middle

    DOLLAR-NEUTRAL: Σ positive weights = 1.0, Σ negative weights = -1.0
    → long notional = short notional = 1.0

    NaN currencies (missing data) are excluded from ranking.
    If fewer than 2*basket_size currencies are available, returns all zeros.
    """
    valid = signal_row.dropna()

    # Need at least 2*basket_size to form non-degenerate baskets
    if len(valid) < 2 * basket_size:
        return pd.Series(0.0, index=signal_row.index)

    ranked = valid.rank(ascending=True, method="first")
    n_valid = len(valid)
    weights = pd.Series(0.0, index=signal_row.index)

    for ccy in valid.index:
        r = ranked[ccy]
        if r <= basket_size:
            # bottom-basket_size → short
            weights[ccy] = -1.0 / basket_size
        elif r > n_valid - basket_size:
            # top-basket_size → long
            weights[ccy] = +1.0 / basket_size
        # else: middle → 0

    return weights


def build_portfolio(
    prices_period: pd.DataFrame,
    k_periods: int,
    basket_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the weight time-series and period return series for the portfolio.

    Returns:
        weights_df : (T, 8) weight matrix (positions held during each period)
        port_rets  : (T,) portfolio net-of-cost returns

    Anti-look-ahead discipline:
        At the close of period t, we observe the trailing K-period signal.
        We THEN set weights that apply DURING period t+1.
        → weights.iloc[t+1] = f(signal.iloc[t])
        → port_ret[t+1] = weights[t+1] . returns[t+1]

    Cost:
        Charged for the fraction of notional that actually turns over
        between period t and period t+1.
        cost_fraction(ccy) = |w_t+1(ccy) - w_t(ccy)| * cost_per_unit(ccy)
        cost_per_unit(ccy) = COST_PIPS * PIP_SIZE[ccy] / PRICE_APPROX[ccy]
    """
    returns = compute_returns(prices_period)
    raw_signal = compute_trailing_return(prices_period, k_periods)

    # Signal-to-weight: signal at t → weight applied at t+1
    # We compute weights for every index point t based on signal[t],
    # then shift them so weight[t] is determined by signal[t-1].
    signal_weights_now = raw_signal.apply(
        lambda row: assign_weights(row, basket_size), axis=1
    )
    # Shift forward by 1: weight[t] = assign_weights(signal[t-1])
    weights_df = signal_weights_now.shift(1)
    weights_df = weights_df.fillna(0.0)

    # Compute cost per rebalance
    cost_per_unit = pd.Series({
        ccy: COST_PIPS * PIP_SIZE[ccy] / PRICE_APPROX[ccy]
        for ccy in ALL_CURRENCIES
    })

    # Gross portfolio return (per period, equal-weight L/S)
    gross_ret = (weights_df * returns).sum(axis=1)

    # Turnover cost per period: only charged on CHANGED positions
    weight_change = weights_df.diff().abs()
    # First period after shift has NaN → fill 0 (first position entry costs are not
    # counted here because they represent month 1 "open from flat"; they are negligible
    # vs the full run but we charge them as a conservatism choice by using the first non-NaN
    # weight row as the first change from 0→w, which diff() handles correctly since
    # weights_df.diff() row[0] = row[0] - 0 = row[0] for the first non-flat period).
    weight_change = weight_change.fillna(0.0)

    # Weighted cost: sum across currencies of |Δw| × cost_per_unit
    # cost_per_unit is scalar per currency (same for every t)
    turnover_cost = (weight_change * cost_per_unit).sum(axis=1)

    net_ret = gross_ret - turnover_cost

    return weights_df, net_ret


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_from_returns(
    period_rets: pd.Series,
    periods_per_year: float,
    label: str = "FULL",
) -> dict:
    """
    Compute annualized metrics from a series of per-period net returns.

    period_rets : pd.Series of per-period portfolio returns (net of costs)
    periods_per_year : 12 for monthly, 52 for weekly

    Returns dict with keys: n, total_return, ann_return, ann_vol, sharpe,
    max_dd, hit_rate, avg_period_ret
    """
    # Filter to non-zero-weight periods only (skip burn-in where weights = 0)
    # A "held" period is one where the portfolio had non-zero net return OR
    # the return is exactly 0 due to cancellation (keep all post-warmup periods).
    # We identify the first period with a non-NaN weight as the start.
    rets = period_rets.dropna()

    if len(rets) < 2:
        return {
            "label": label, "n": 0,
            "total_return": None, "ann_return": None,
            "ann_vol": None, "sharpe": None, "max_dd": None,
            "hit_rate": None, "avg_period_ret": None,
        }

    n = len(rets)
    total_return = (1 + rets).prod() - 1
    ann_return   = (1 + total_return) ** (periods_per_year / n) - 1
    ann_vol      = rets.std() * np.sqrt(periods_per_year)
    sharpe       = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown on equity curve
    equity = (1 + rets).cumprod()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = abs(dd.min())

    hit_rate      = (rets > 0).mean()
    avg_period_ret = rets.mean()

    return {
        "label":          label,
        "n":              n,
        "total_return":   total_return,
        "ann_return":     ann_return,
        "ann_vol":        ann_vol,
        "sharpe":         sharpe,
        "max_dd":         max_dd,
        "hit_rate":       hit_rate,
        "avg_period_ret": avg_period_ret,
    }


def split_by_window(
    net_ret: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Split net returns into FULL, IS, OOS sub-series."""
    full_ret = net_ret
    is_ret   = net_ret[(net_ret.index >= IS_START) & (net_ret.index <= IS_END)]
    oos_ret  = net_ret[(net_ret.index >= OOS_START) & (net_ret.index <= OOS_END)]
    return full_ret, is_ret, oos_ret


# ---------------------------------------------------------------------------
# Smoke-test helpers
# ---------------------------------------------------------------------------

def smoke_check_usd_price_series(prices_4h: pd.DataFrame):
    """
    Print a quick sanity check on the 8 USD-denominated price series.

    Expected patterns (approximate, per scope doc):
    - EUR: ends different from start (various % over 6y)
    - JPY: weakens 2022-24 (USD/JPY from ~115 to ~155 → inverted = 1/JPY falls)
    - USD: constant 1.0 throughout
    - CAD, CHF: vary modestly
    """
    print("\n--- SMOKE: Per-currency USD-price sanity ---")
    monthly = resample_to_month_end(prices_4h)

    # Monthly bar counts — check continuous coverage
    print(f"\n  Monthly bar counts (expect ~77-78 bars per currency):")
    for ccy in ALL_CURRENCIES:
        if ccy in monthly.columns:
            non_null = monthly[ccy].notna().sum()
            first = monthly[ccy].first_valid_index()
            last  = monthly[ccy].last_valid_index()
            print(f"    {ccy:5s}  bars={non_null:3d}  "
                  f"first={first.strftime('%Y-%m') if first else 'N/A'}  "
                  f"last={last.strftime('%Y-%m') if last else 'N/A'}")

    # Price levels at a few anchor dates
    # IMPORTANT: monthly.loc["2020-01"] on a DatetimeIndex returns a *DataFrame*
    # (all rows in that month), not a Series — we explicitly select a scalar row
    # using period-matching and .squeeze() so downstream .get(ccy) works correctly.
    print(f"\n  Price snapshots (USD per 1 unit of CCY):")
    anchor_dates = ["2020-01", "2022-01", "2022-12", "2023-12", "2026-05"]
    for anchor in anchor_dates:
        try:
            period = pd.Period(anchor, "M")
            subset = monthly[monthly.index.to_period("M") == period]
            if subset.empty:
                row = None
            else:
                row = subset.iloc[-1]  # Series (last row in that month)
        except Exception:
            row = None
        if row is not None:
            vals = "  ".join(
                f"{ccy}={float(row[ccy]):.4f}" if (ccy in row.index and pd.notna(row[ccy])) else f"{ccy}=N/A"
                for ccy in ALL_CURRENCIES
            )
            print(f"    {anchor}: {vals}")

    # Total return 2020→2026 per currency (as the signal would see it)
    first_prices = monthly.dropna(how="all").iloc[0]
    last_prices  = monthly.dropna(how="all").iloc[-1]
    total_rets = (last_prices / first_prices - 1) * 100
    print(f"\n  Total % return 2020→2026 per currency (USD-denominated):")
    for ccy in ALL_CURRENCIES:
        if ccy in total_rets.index:
            print(f"    {ccy:5s}: {total_rets[ccy]:+7.2f}%")
    print()


def smoke_check_dollar_neutrality(weights_df: pd.DataFrame):
    """
    Assert dollar-neutrality: at every rebalance, long notional == short notional.
    Prints the result; raises on material violation.
    """
    # Only check rows where the portfolio is non-trivially invested
    w = weights_df[(weights_df != 0).any(axis=1)]
    long_sums  = w.clip(lower=0).sum(axis=1)
    short_sums = w.clip(upper=0).abs().sum(axis=1)

    diff = (long_sums - short_sums).abs()
    max_imbalance = diff.max()

    print(f"\n--- SMOKE: Dollar-neutrality check ---")
    print(f"  Max long/short notional imbalance across {len(w)} invested periods: "
          f"{max_imbalance:.6f}")
    if max_imbalance > 1e-9:
        print(f"  WARNING: non-zero imbalance detected!")
    else:
        print(f"  PASS: dollar-neutral at every rebalance.")


def smoke_check_usd_in_long_2022(weights_df: pd.DataFrame):
    """
    Verify: in 2022 (year of USD dominance), USD should rank near the top
    and appear in the long basket for most months.
    Scope doc: 'in 2022–23 USD was itself a high-carry, high-momentum currency'.
    """
    w_2022 = weights_df[
        (weights_df.index.year == 2022) & ("USD" in weights_df.columns)
    ]
    if w_2022.empty:
        print("  SMOKE/2022-USD: no 2022 rows to check.")
        return

    usd_long  = (w_2022["USD"] > 0).sum()
    usd_short = (w_2022["USD"] < 0).sum()
    usd_flat  = (w_2022["USD"] == 0).sum()
    print(f"\n--- SMOKE: USD positioning in 2022 ---")
    print(f"  USD long: {usd_long} months, short: {usd_short}, flat: {usd_flat}")
    if usd_long > usd_short:
        print("  PASS: USD appears in long basket more often than short in 2022.")
    else:
        print("  NOTE: USD not predominantly long in 2022 for K=3 monthly. "
              "This can vary by lookback; check 2022 with K=1 or K=6 if concerned.")


def smoke_check_rebalance_count(net_ret: pd.Series, freq: str):
    """Confirm rebalance count is in expected range."""
    n = len(net_ret.dropna())
    if freq == "monthly":
        expected = (78, 78)  # 2020-01 to 2026-06 ≈ 78 months
    else:
        expected = (330, 340)  # ~6.5y × 52 ≈ 338 weeks
    print(f"\n--- SMOKE: Rebalance count ({freq}) ---")
    print(f"  N periods in return series: {n}  (expected approx {expected[0]}–{expected[1]})")
    if expected[0] - 10 <= n <= expected[1] + 10:
        print("  PASS: rebalance count in plausible range.")
    else:
        print("  WARNING: unexpected rebalance count — check alignment.")


def smoke_check_sanity_sharpe(sharpe: Optional[float], freq: str, k: int, basket: int):
    """Flag suspiciously high Sharpe."""
    print(f"\n--- SMOKE: Sharpe sanity (K={k}, basket={basket}, {freq}) ---")
    if sharpe is None:
        print("  Sharpe: N/A")
        return
    print(f"  Sharpe: {sharpe:.3f}")
    if sharpe > 3.0:
        print("  WARNING: Sharpe > 3.0 on monthly L/S FX probe is a red flag. "
              "Check look-ahead alignment (signal at t vs return at t+1).")
    elif sharpe > 1.5:
        print("  NOTE: Sharpe > 1.5 — plausible but check OOS before trusting.")
    else:
        print("  PASS: Sharpe in plausible range.")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_pct(v, digits=1):
    if v is None:
        return "   N/A "
    return f"{v*100:+7.{digits}f}%"


def fmt_f(v, fmt=".3f"):
    if v is None:
        return "   N/A "
    return format(v, fmt)


def fmt_n(v):
    if v is None:
        return "  N/A"
    return f"{v:5d}"


def print_metrics_row(label: str, m: dict, prefix: str = "  "):
    """Print one row of the results table."""
    if m["n"] == 0:
        print(f"{prefix}{label:<8}  N=   0   (no data in window)")
        return
    print(
        f"{prefix}{label:<8}  "
        f"N={m['n']:4d}  "
        f"TotRet={fmt_pct(m['total_return'])}  "
        f"AnnRet={fmt_pct(m['ann_return'])}  "
        f"Vol={fmt_pct(m['ann_vol'])}  "
        f"Sharpe={fmt_f(m['sharpe'])}  "
        f"MaxDD={fmt_pct(m['max_dd'], 1)}  "
        f"HitRate={fmt_pct(m['hit_rate'], 1)}  "
        f"AvgPRet={fmt_pct(m['avg_period_ret'], 3)}"
    )


def print_verdict_row(
    k: int,
    basket: int,
    freq: str,
    oos_m: dict,
) -> bool:
    """Evaluate the pre-registered GO/NO-GO for one config; return True if GO."""
    if oos_m["n"] < 5:
        verdict = "SKIP (insufficient OOS data)"
        go = False
    else:
        oos_sharpe = oos_m["sharpe"] or 0.0
        oos_dd     = oos_m["max_dd"] or 1.0
        oos_ret    = oos_m["total_return"] or -1.0

        sharpe_ok = oos_sharpe >= OOS_SHARPE_MIN
        dd_ok     = oos_dd     <= OOS_MAX_DD_MAX
        ret_ok    = oos_ret    >  OOS_TOTAL_RET_MIN

        go = sharpe_ok and dd_ok and ret_ok

        reasons = []
        if not sharpe_ok:
            reasons.append(f"OOS Sharpe {oos_sharpe:.3f} < {OOS_SHARPE_MIN}")
        if not dd_ok:
            reasons.append(f"OOS MaxDD {oos_dd*100:.1f}% > {OOS_MAX_DD_MAX*100:.0f}%")
        if not ret_ok:
            reasons.append(f"OOS TotRet {oos_ret*100:.1f}% <= 0")

        if go:
            verdict = "GO"
        else:
            verdict = "NO-GO  [" + "; ".join(reasons) + "]"

    tag = f"K={k:2d}  basket={basket}  {freq:8s}"
    marker = "***" if go else "   "
    print(f"  {marker} {tag}  OOS: {verdict}")
    return go


# ---------------------------------------------------------------------------
# Main probe runner
# ---------------------------------------------------------------------------

def run_probe(smoke_only: bool = False):

    print("=" * 80)
    print("FX CROSS-SECTIONAL MOMENTUM PROBE — Pre-registered experiment")
    print(f"Universe: {', '.join(ALL_CURRENCIES)}")
    print(f"Window:   {WINDOW_START} → {WINDOW_END}")
    print(f"IS/OOS:   IS={IS_START}–{IS_END} / OOS={OOS_START}–{OOS_END}")
    print(f"K-months: {K_MONTHS}")
    print(f"Baskets:  {BASKET_SIZES}  (top-K long, bottom-K short, equal-weight)")
    print(f"Freqs:    {REBALANCE_FREQS}")
    print(f"Cost:     {COST_PIPS} pips per leg per changed position")
    print(f"GO bar (pre-registered): OOS Sharpe >= {OOS_SHARPE_MIN}, "
          f"MaxDD <= {OOS_MAX_DD_MAX*100:.0f}%, TotRet > 0")
    if smoke_only:
        print("MODE: SMOKE TEST — K=3, basket=2, monthly only. Full job NOT run.")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------
    # Load 4H candles once (bulk load, ~6.5 years × 7 pairs)
    # -------------------------------------------------------------------
    print("Loading 4H candles from ig_candles_backtest...")
    conn = get_connection()

    pair_closes_raw: Dict[str, pd.Series] = {}
    for ccy, epic, invert in PAIRS:
        print(f"  {ccy:5s} ({epic})...", end=" ", flush=True)
        df = load_4h_candles(conn, epic)
        if df.empty:
            print("ERROR: no data returned")
        else:
            pair_closes_raw[ccy] = df["close"]
            print(f"{len(df):,} 4H bars  "
                  f"({df.index[0].strftime('%Y-%m-%d')} → "
                  f"{df.index[-1].strftime('%Y-%m-%d')})")

    conn.close()

    if len(pair_closes_raw) < 7:
        print(f"ABORT: only {len(pair_closes_raw)} of 7 pairs loaded. Check ig_candles_backtest coverage.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Build 8 USD-denominated price series at 4H
    # -------------------------------------------------------------------
    prices_4h = build_usd_price_series(pair_closes_raw)
    print(f"\n4H price DataFrame: {prices_4h.shape[0]} bars × {prices_4h.shape[1]} currencies")

    # -------------------------------------------------------------------
    # Smoke test: price series sanity
    # -------------------------------------------------------------------
    smoke_check_usd_price_series(prices_4h)

    if smoke_only:
        # Run one config only: K=3, basket=2, monthly
        smoke_k       = 3
        smoke_basket  = 2
        smoke_freq    = "monthly"

        print("=" * 80)
        print(f"SMOKE CONFIG: K={smoke_k}, basket={smoke_basket}, {smoke_freq}")
        print("=" * 80)

        prices_period = resample_to_month_end(prices_4h)
        k_periods = smoke_k  # monthly K=3 → 3 monthly bars

        weights_df, net_ret = build_portfolio(prices_period, k_periods, smoke_basket)

        # Trim leading periods where no position (signal warm-up)
        first_active = (weights_df != 0).any(axis=1).idxmax()
        net_ret_trimmed = net_ret[net_ret.index >= first_active]

        smoke_check_dollar_neutrality(weights_df)
        smoke_check_usd_in_long_2022(weights_df)
        smoke_check_rebalance_count(net_ret_trimmed, smoke_freq)

        full_ret, is_ret, oos_ret = split_by_window(net_ret_trimmed)

        m_full = compute_metrics_from_returns(full_ret, 12.0, "FULL")
        m_is   = compute_metrics_from_returns(is_ret,  12.0, "IS")
        m_oos  = compute_metrics_from_returns(oos_ret, 12.0, "OOS")

        print("\n  Results for smoke config:")
        print_metrics_row("FULL", m_full)
        print_metrics_row("IS",   m_is)
        print_metrics_row("OOS",  m_oos)

        smoke_check_sanity_sharpe(m_full["sharpe"], smoke_freq, smoke_k, smoke_basket)

        print()
        print("  Sample of weights (first 5 invested months):")
        invested_months = weights_df[(weights_df != 0).any(axis=1)].head(5)
        for ts, row in invested_months.iterrows():
            longs  = [c for c in ALL_CURRENCIES if row.get(c, 0) > 0]
            shorts = [c for c in ALL_CURRENCIES if row.get(c, 0) < 0]
            print(f"    {ts.strftime('%Y-%m')}: LONG {longs}  SHORT {shorts}")

        print()
        print("*** SMOKE TEST COMPLETE — no full grid run, no file written. ***")
        print("    To run the full job: python /app/fx_xsec_momentum_probe.py > /tmp/fx_xsec_results.txt 2>&1")
        print("=" * 80)
        return

    # -------------------------------------------------------------------
    # Full grid run
    # -------------------------------------------------------------------
    print("=" * 80)
    print("FULL GRID RUN")
    print("=" * 80)

    # Pre-compute monthly and weekly price series once
    prices_monthly = resample_to_month_end(prices_4h)
    prices_weekly  = resample_to_week_end(prices_4h)

    # Store all results for the verdict table at the end
    all_results = []  # list of (k, basket, freq, m_full, m_is, m_oos)

    for freq in REBALANCE_FREQS:
        periods_per_year = 12.0 if freq == "monthly" else 52.0
        prices_period = prices_monthly if freq == "monthly" else prices_weekly

        for k in K_MONTHS:
            # Convert K months to number of periods for weekly
            # K=3 months ≈ 13 weeks; K=12 months ≈ 52 weeks, etc.
            # Use pandas DateOffset month-to-period mapping:
            # We shift price series by K months directly using DateOffset shift
            # on the closed-form signal. For weekly, convert months → weeks.
            if freq == "monthly":
                k_periods = k  # exactly K monthly bars
            else:
                # Convert K months to weeks: use 4.3478 weeks/month
                k_periods = max(1, round(k * 4.3478))

            for basket in BASKET_SIZES:
                print(f"\n  Running K={k:2d}mo  basket={basket}  {freq}  "
                      f"(k_periods={k_periods})...", end=" ", flush=True)

                weights_df, net_ret = build_portfolio(
                    prices_period, k_periods, basket
                )

                # Trim warm-up (first k_periods bars have no valid signal)
                first_active = (weights_df != 0).any(axis=1)
                if not first_active.any():
                    print("no active periods — skip")
                    continue
                first_ts = first_active.idxmax()
                net_ret_active = net_ret[net_ret.index >= first_ts]

                full_ret, is_ret, oos_ret = split_by_window(net_ret_active)

                m_full = compute_metrics_from_returns(full_ret, periods_per_year, "FULL")
                m_is   = compute_metrics_from_returns(is_ret,  periods_per_year, "IS")
                m_oos  = compute_metrics_from_returns(oos_ret, periods_per_year, "OOS")

                all_results.append((k, basket, freq, m_full, m_is, m_oos))
                print(f"done (N_full={m_full['n']}, N_oos={m_oos['n']})")

    # -------------------------------------------------------------------
    # Print full results table
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)

    current_freq = None
    for k, basket, freq, m_full, m_is, m_oos in all_results:
        if freq != current_freq:
            print(f"\n{'='*60}")
            print(f"  REBALANCE: {freq.upper()}")
            print(f"{'='*60}")
            current_freq = freq

        print(f"\n  K={k:2d}mo  basket_size={basket} "
              f"(long top-{basket}, short bottom-{basket})")
        print_metrics_row("FULL", m_full)
        print_metrics_row("IS  ", m_is)
        print_metrics_row("OOS ", m_oos)

    # -------------------------------------------------------------------
    # Pre-registered verdict
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT TABLE (pre-registered bar: OOS Sharpe >= 0.5, MaxDD <= 25%, TotRet > 0)")
    print("=" * 80)

    any_go = False
    for k, basket, freq, m_full, m_is, m_oos in all_results:
        go = print_verdict_row(k, basket, freq, m_oos)
        if go:
            any_go = True

    # -------------------------------------------------------------------
    # Final decision
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("FINAL DECISION")
    print("=" * 80)
    if any_go:
        print()
        print("  GO: At least one config meets the pre-registered OOS bar.")
        print("  → This cross-sectional momentum result warrants further analysis.")
        print("  → Step 2 in scope doc: source rate data + CARRY probe.")
        print("  → Temper expectations: G10-only, built from same return series as")
        print("    killed time-series momentum — relative ranking removes common USD")
        print("    factor but is a recombination of dead material. (Scope doc §'Recommended')")
        print("  → DO NOT add filters to marginal configs that just missed the bar.")
    else:
        print()
        print("  NO-GO: No config meets the pre-registered OOS bar.")
        print("  Cross-sectional momentum does not survive OOS on G10 at this cost (2 pips).")
        print("  → Consistent with scope doc expectation: 'G10-only FX momentum is")
        print("    documented as weak and cost-sensitive. Don't be surprised if NO-GO.'")
        print("  → Do not proceed to carry probe on the basis of momentum alone.")
        print("  → Consider whether carry (documented stronger prior) warrants a separate")
        print("    probe despite momentum failing — scope doc allows this.")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FX Cross-Sectional Momentum Probe (pre-registered)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke test only: run K=3, basket=2, monthly on full 2020-2026 window. "
            "Prints sanity checks and stops. Does NOT run the full K×basket×freq grid."
        ),
    )
    args = parser.parse_args()
    run_probe(smoke_only=args.smoke)
