#!/usr/bin/env python3
"""
Scan daily stock candles for the "downtrend -> sideways base -> climb" pattern
(rounding-bottom / accumulation-then-recovery), as seen on the ANAB daily chart.

Phases (temporal order, ending at the most recent candle), anchored on the
climb-start pivot (the base low that price has since turned up off of):

  A) DOWNTREND : price falls meaningfully from a prior peak into the base
  B) BASE      : >=BASE_MIN candles of low-volatility consolidation (flat, tight range)
  C) CLIMB     : recent rise off the base low, not yet fully retraced to the peak

Everything is measured in % / returns. Liquidity + recency floors applied.
"""
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DB = os.environ["STOCKS_DATABASE_URL"]
LOOKBACK_DAYS = 50          # daily candles pulled per ticker
MAX_STALE_DAYS = 4          # last candle must be within this many days of the global max date

# --- Pattern thresholds (tunable) ---
MIN_PRICE        = 3.0      # liquidity floor: current close
MIN_DOLLAR_VOL   = 1_000_000  # median daily $-volume over base+climb window
DECLINE_MIN      = 0.10     # downtrend: >=10% drop from peak to base low
BASE_MIN_LEN     = 4        # base: minimum consolidation candles
BASE_MAX_LEN     = 18       # base: maximum consolidation candles considered
BASE_RANGE_MAX   = 0.07     # base: (max_close-min_close)/mean_close must be tighter than this
CLIMB_MIN        = 0.04     # climb: >=4% up off the base low
CLIMB_MAX_BARS   = 8        # climb started within the last N candles
CLIMB_MIN_BARS   = 2        # climb is at least 2 candles old (not a 1-day spike)
RETRACE_MAX      = 0.90     # climb must not have retraced >90% of the decline (else it's a late/round-trip setup)
DECLINE_MAX      = 0.60     # backstop: reject improbably deep declines (likely split/data artifacts)
MAX_BAR_MOVE     = 0.50     # split/artifact guard: reject if any single daily move exceeds this

engine = create_engine(DB)

print("Loading daily candles...")
q = text("""
    WITH ranked AS (
        SELECT ticker, timestamp::date AS d, open, high, low, close, volume,
               row_number() OVER (PARTITION BY ticker ORDER BY timestamp DESC) AS rn
        FROM stock_candles_synthesized
        WHERE timeframe = '1d'
    )
    SELECT ticker, d, open, high, low, close, volume
    FROM ranked WHERE rn <= :lb
""")
df = pd.read_sql(q, engine, params={"lb": LOOKBACK_DAYS})
df["close"] = df["close"].astype(float)
df["high"] = df["high"].astype(float)
df["low"] = df["low"].astype(float)
df["volume"] = df["volume"].fillna(0).astype(float)
global_max = df["d"].max()
print(f"  {df['ticker'].nunique()} tickers, max date {global_max}")


def analyze(g: pd.DataFrame):
    g = g.sort_values("d").reset_index(drop=True)
    n = len(g)
    if n < 20:
        return None
    last_date = g["d"].iloc[-1]
    if (global_max - last_date).days > MAX_STALE_DAYS:
        return None  # stale ticker

    close = g["close"].to_numpy(dtype=float)
    last = close[-1]
    if last < MIN_PRICE:
        return None

    # --- Split / data-artifact guard: no single daily move beyond MAX_BAR_MOVE ---
    bar_moves = np.abs(np.diff(close) / close[:-1])
    if bar_moves.max() > MAX_BAR_MOVE:
        return None

    # --- Locate climb-start pivot = lowest close in the recent window ---
    recent_win = min(CLIMB_MAX_BARS + 2, n - 1)
    tail = close[-recent_win:]
    trough_off = int(np.argmin(tail))               # offset within tail
    trough_idx = n - recent_win + trough_off         # absolute index of base low
    bars_since_trough = (n - 1) - trough_idx
    if not (CLIMB_MIN_BARS <= bars_since_trough <= CLIMB_MAX_BARS):
        return None
    trough = close[trough_idx]
    if trough <= 0:
        return None

    # --- CLIMB: rise off the base low, and currently still near the highs of the climb ---
    climb = (last - trough) / trough
    if climb < CLIMB_MIN:
        return None
    # must be genuinely rising, not a flat drift: last close near the climb's max
    climb_seg = close[trough_idx:]
    if last < climb_seg.max() * 0.985:
        return None

    # --- BASE: consolidation window ending at the trough ---
    # Among all qualifying windows, keep the one that makes the strongest case
    # (best combined tightness + flatness + length), not merely the longest.
    clamp = lambda v: max(0.0, min(1.0, v))
    best_base = None
    best_base_q = -1.0
    for blen in range(BASE_MIN_LEN, BASE_MAX_LEN + 1):
        b_start = trough_idx - blen + 1
        if b_start < 1:
            break
        seg = close[b_start:trough_idx + 1]
        rng = (seg.max() - seg.min()) / seg.mean()
        if rng > BASE_RANGE_MAX:
            continue
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0] / seg.mean()       # normalized per-bar slope
        q = clamp(1 - rng / BASE_RANGE_MAX) + clamp(1 - abs(slope) / 0.006) + clamp(blen / 8.0)
        if q > best_base_q:
            best_base_q = q
            best_base = {"len": blen, "start": b_start, "range": rng,
                         "slope": slope, "level": float(seg.mean())}
    if best_base is None:
        return None
    base_level = best_base["level"]
    base_start = best_base["start"]

    # --- DOWNTREND: peak before the base down into it ---
    pre = close[:base_start]
    if len(pre) < 3:
        return None
    peak_idx = int(np.argmax(pre))
    peak = pre[peak_idx]
    decline = (peak - base_level) / peak
    if decline < DECLINE_MIN or decline > DECLINE_MAX:
        return None
    # peak must precede the base (it does by construction) and downtrend must dominate:
    # base level should sit in the lower portion of the peak->trough range
    if base_level > peak * (1 - DECLINE_MIN / 2):
        return None

    # --- Retrace guard: climb shouldn't have round-tripped back to the peak ---
    retrace = (last - trough) / (peak - trough) if peak > trough else 1.0
    if retrace > RETRACE_MAX:
        return None

    # --- Liquidity over base+climb window ---
    win = g.iloc[base_start:]
    dvol = (win["close"] * win["volume"]).median()
    if dvol < MIN_DOLLAR_VOL:
        return None

    # --- Fit score: six normalized [0,1] subscores, centered on the canonical shape ---
    s_decline = clamp(1 - abs(decline - 0.25) / 0.30)        # ideal ~25% drop (triangular)
    s_base_tight = clamp(1 - best_base["range"] / BASE_RANGE_MAX)
    s_base_flat = clamp(1 - abs(best_base["slope"]) / 0.006)
    s_base_len = clamp(best_base["len"] / 8.0)
    s_retrace = clamp(1 - abs(retrace - 0.50) / 0.50)        # ideal ~halfway back up
    s_climb = clamp(climb / 0.10)                            # healthy climb magnitude, capped
    score = s_decline + s_base_tight + s_base_flat + s_base_len + s_retrace + s_climb

    return {
        "ticker": g["ticker"].iloc[0],
        "score": round(score, 3),
        "last": round(last, 2),
        "decline_pct": round(decline * 100, 1),
        "peak": round(peak, 2),
        "base_level": round(base_level, 2),
        "base_len": best_base["len"],
        "base_range_pct": round(best_base["range"] * 100, 1),
        "trough": round(trough, 2),
        "climb_pct": round(climb * 100, 1),
        "climb_bars": bars_since_trough,
        "retrace_pct": round(retrace * 100, 0),
        "dvol_m": round(dvol / 1e6, 1),
        "last_date": str(last_date),
    }


results = []
for ticker, g in df.groupby("ticker"):
    r = analyze(g)
    if r:
        results.append(r)

res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
print(f"\n=== {len(res)} matches (downtrend -> base -> climb) ===\n")
cols = ["ticker", "score", "last", "decline_pct", "peak", "base_level", "base_len",
        "base_range_pct", "trough", "climb_pct", "climb_bars", "retrace_pct", "dvol_m"]
pd.set_option("display.width", 200, "display.max_columns", 30)
print(res[cols].head(40).to_string(index=False))

# Ground-truth: where does ANAB rank?
if "ANAB" in set(res["ticker"]):
    rank = res.index[res["ticker"] == "ANAB"][0] + 1
    print(f"\n[ground-truth] ANAB ranks #{rank} of {len(res)}")
    print(res[res["ticker"] == "ANAB"][cols].to_string(index=False))
else:
    print("\n[ground-truth] WARNING: ANAB did NOT match — thresholds need adjustment")
