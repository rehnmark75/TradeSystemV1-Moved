"""
Decline -> Base -> Climb (rounding-bottom) daily pattern detector.
==================================================================
Single source of truth for the detector. Pure numpy: no DB, no pandas — safe
to import in the live scanner save path.

Three phases in temporal order, ending at the most recent COMPLETED bar
(the as-of bar):
  A) DOWNTREND : >=10% drop from a prior peak into the base (<=60% guards splits)
  B) BASE      : >=4 candles of tight (<7% range), ~flat consolidation
  C) CLIMB     : rising off the base low (>=4%), started 2-8 bars ago, not yet
                 fully retraced to the peak

Plus liquidity floor + a single-bar split/data-artifact guard. Returns
(matched: bool, score: float in [0..6]).

Validated against ANAB (#50/332 on 2026-06-16) and used by:
  - analysis/scan_decline_base_climb.py   (universe scan)
  - analysis/pattern_overlap_edge.py      (overlap + forward-edge harness)
  - scanners/base_scanner.py              (monitor-only signal flagging)

Thresholds here are AUTHORITATIVE — changing them invalidates the in-sample
gap_and_go baseline (PF 0.89 -> 1.21). Bump PATTERN_VERSION on any change.
"""
import numpy as np

PATTERN_VERSION = "dbc-1.0"

# --- Pattern thresholds (authoritative) ---
LOOKBACK_DAYS    = 50         # max daily bars considered
MIN_PRICE        = 3.0        # liquidity floor: last close
MIN_DOLLAR_VOL   = 1_000_000  # median daily $-volume over base+climb window
DECLINE_MIN      = 0.10       # downtrend: >=10% peak->base drop
DECLINE_MAX      = 0.60       # backstop: reject improbably deep drops (splits)
BASE_MIN_LEN     = 4
BASE_MAX_LEN     = 18
BASE_RANGE_MAX   = 0.07       # base tightness: (max-min)/mean
CLIMB_MIN        = 0.04       # climb: >=4% off base low
CLIMB_MAX_BARS   = 8          # climb started within last N bars
CLIMB_MIN_BARS   = 2          # climb at least 2 bars old (not a 1-day spike)
RETRACE_MAX      = 0.90       # not fully round-tripped to the peak
MAX_BAR_MOVE     = 0.50       # split/artifact guard: max single-bar move

_clamp = lambda v: max(0.0, min(1.0, v))


def detect_pattern(close, high, low, volume):
    """Detect the pattern on arrays ending at the as-of (most recent completed)
    bar. Arrays must be ascending; only `close` and `volume` are used today
    (high/low accepted for signature stability). Returns (matched, score)."""
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    n = len(close)
    if n < 20:
        return False, 0.0
    if not (np.isfinite(close).all() and np.isfinite(volume).all()):
        return False, 0.0   # stray nan OHLC/volume bar -> skip
    last = close[-1]
    if last < MIN_PRICE:
        return False, 0.0
    bar_moves = np.abs(np.diff(close) / close[:-1])
    if bar_moves.max() > MAX_BAR_MOVE:
        return False, 0.0

    # Climb-start pivot = lowest close in the recent window
    recent_win = min(CLIMB_MAX_BARS + 2, n - 1)
    tail = close[-recent_win:]
    trough_off = int(np.argmin(tail))
    trough_idx = n - recent_win + trough_off
    bars_since_trough = (n - 1) - trough_idx
    if not (CLIMB_MIN_BARS <= bars_since_trough <= CLIMB_MAX_BARS):
        return False, 0.0
    trough = close[trough_idx]
    if trough <= 0:
        return False, 0.0

    # CLIMB
    climb = (last - trough) / trough
    if climb < CLIMB_MIN:
        return False, 0.0
    climb_seg = close[trough_idx:]
    if last < climb_seg.max() * 0.985:
        return False, 0.0

    # BASE: pick the window making the strongest case (tight + flat + length)
    best_base = None
    best_q = -1.0
    for blen in range(BASE_MIN_LEN, BASE_MAX_LEN + 1):
        b_start = trough_idx - blen + 1
        if b_start < 1:
            break
        seg = close[b_start:trough_idx + 1]
        rng = (seg.max() - seg.min()) / seg.mean()
        if rng > BASE_RANGE_MAX:
            continue
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0] / seg.mean()
        q = _clamp(1 - rng / BASE_RANGE_MAX) + _clamp(1 - abs(slope) / 0.006) + _clamp(blen / 8.0)
        if q > best_q:
            best_q = q
            best_base = {"len": blen, "start": b_start, "range": rng,
                         "slope": slope, "level": float(seg.mean())}
    if best_base is None:
        return False, 0.0
    base_level = best_base["level"]
    base_start = best_base["start"]

    # DOWNTREND
    pre = close[:base_start]
    if len(pre) < 3:
        return False, 0.0
    peak = pre[int(np.argmax(pre))]
    decline = (peak - base_level) / peak
    if decline < DECLINE_MIN or decline > DECLINE_MAX:
        return False, 0.0
    if base_level > peak * (1 - DECLINE_MIN / 2):
        return False, 0.0

    # Retrace guard
    retrace = (last - trough) / (peak - trough) if peak > trough else 1.0
    if retrace > RETRACE_MAX:
        return False, 0.0

    # Liquidity
    win_vol = volume[base_start:]
    win_cls = close[base_start:]
    dvol = float(np.median(win_cls * win_vol))
    if dvol < MIN_DOLLAR_VOL:
        return False, 0.0

    # Fit score: six normalized [0,1] subscores centered on the canonical shape
    s_decline = _clamp(1 - abs(decline - 0.25) / 0.30)
    s_base_tight = _clamp(1 - best_base["range"] / BASE_RANGE_MAX)
    s_base_flat = _clamp(1 - abs(best_base["slope"]) / 0.006)
    s_base_len = _clamp(best_base["len"] / 8.0)
    s_retrace = _clamp(1 - abs(retrace - 0.50) / 0.50)
    s_climb = _clamp(climb / 0.10)
    score = s_decline + s_base_tight + s_base_flat + s_base_len + s_retrace + s_climb
    return True, round(float(score), 3)
