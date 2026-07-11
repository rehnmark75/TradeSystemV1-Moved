"""Auto-Pause Layer — configuration defaults and DSN helpers (Phase 1).

All thresholds are FIXED, first-principles defaults chosen for *generalization*
across strategies — they were validated cross-strategy, NOT tuned to maximise
SEK on any single decay episode. Overriding via env vars is fine for
experimentation; do not re-tune them on historical data, which would
reintroduce the single-regime overfit the whole design exists to avoid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def forex_dsn() -> str:
    """DSN for the forex DB (trade_log, alert_history, ig_candles)."""
    return os.getenv(
        "DATABASE_URL",
        os.getenv(
            "TRADING_DSN",
            "host=postgres dbname=forex user=postgres password=postgres",
        ),
    )


def strategy_config_dsn() -> str:
    """DSN for the strategy_config DB (pair_overrides, auto_pause_eligibility)."""
    return os.getenv(
        "STRATEGY_CONFIG_DATABASE_URL",
        "postgresql://postgres:postgres@postgres:5432/strategy_config",
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AutoPauseParams:
    """Fixed trip/resume thresholds. See module docstring on tuning."""

    # --- Trip Rule A (trade-based; pause -> monitor-only) ---
    rolling_window: int = 10              # last-N closed trades used for PF
    trip_pf_threshold: float = 0.8        # pause if rolling PF < this ...
    trip_min_trades: int = 10             # ... and at least this many closed trades
    trip_max_consecutive_losses: int = 5  # universal safeguard (fires at low freq too)

    # --- Trip Rule B (shadow ref-grid series from monitor_only_outcomes) ---
    # Ref-grid PF is near-binary (+tp/-sl, breakeven WR 40% at RR 1.5), so the
    # rule requires BOTH a PF floor and a WR drop vs the frozen enrollment
    # baseline; sigma(WR) ~ 7pp at window 50, so the 12pp drop is ~1.7 sigma.
    shadow_window: int = 50                     # last-N RESOLVED shadow outcomes
    shadow_min_outcomes: int = 30               # below this, no PF/WR trip
    shadow_trip_pf: float = 0.8                 # absolute PF floor
    shadow_trip_wr_drop: float = 0.12           # WR must be this far below baseline
    shadow_max_consecutive_losses: int = 8      # safeguard (5 is common at WR ~50%)

    # --- Enforcement mode ---
    # Dry run: trips are logged + recorded as events but monitor_only is NOT
    # flipped. Default ON so a fresh deployment observes for ~a week first.
    dry_run: bool = True

    # --- Eligibility floor ---
    min_monthly_trades: float = 8.0       # below this a cell is OUT OF SCOPE (rolling
                                          # PF never fills in time; resume could take months)

    # --- Resume (Phase 3 — kept here as the single source of truth; unused in Phase 1) ---
    resume_pf_threshold: float = 1.1      # hysteresis gap ABOVE trip (0.8) to stop flip-flop
    resume_min_signals: int = 15          # fresh reconstructed outcomes required post-pause
    resume_cooldown_days: int = 10


def default_params() -> AutoPauseParams:
    """Build params from env overrides, falling back to the fixed defaults."""
    return AutoPauseParams(
        rolling_window=_env_int("AUTO_PAUSE_WINDOW", 10),
        trip_pf_threshold=_env_float("AUTO_PAUSE_TRIP_PF", 0.8),
        trip_min_trades=_env_int("AUTO_PAUSE_MIN_TRADES", 10),
        trip_max_consecutive_losses=_env_int("AUTO_PAUSE_MAX_CONSEC_LOSSES", 5),
        shadow_window=_env_int("AUTO_PAUSE_SHADOW_WINDOW", 50),
        shadow_min_outcomes=_env_int("AUTO_PAUSE_SHADOW_MIN_OUTCOMES", 30),
        shadow_trip_pf=_env_float("AUTO_PAUSE_SHADOW_TRIP_PF", 0.8),
        shadow_trip_wr_drop=_env_float("AUTO_PAUSE_SHADOW_TRIP_WR_DROP", 0.12),
        shadow_max_consecutive_losses=_env_int("AUTO_PAUSE_SHADOW_MAX_CONSEC_LOSSES", 8),
        dry_run=os.getenv("AUTO_PAUSE_DRY_RUN", "true").strip().lower()
        not in ("false", "0", "no", "off"),
        min_monthly_trades=_env_float("AUTO_PAUSE_MIN_MONTHLY_TRADES", 8.0),
        resume_pf_threshold=_env_float("AUTO_PAUSE_RESUME_PF", 1.1),
        resume_min_signals=_env_int("AUTO_PAUSE_RESUME_MIN_SIGNALS", 15),
        resume_cooldown_days=_env_int("AUTO_PAUSE_RESUME_COOLDOWN_DAYS", 10),
    )
