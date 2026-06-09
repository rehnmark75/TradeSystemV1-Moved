"""Auto-Pause Layer (Phase 1 building blocks).

A per-(strategy, pair) layer that auto-flips a cell to monitor-only when its
rolling LIVE performance decays, and (later phases) auto-resumes it when shadow
performance recovers. This package contains the standalone, testable building
blocks; it does NOT yet hook into the live scan loop (that is Phase 2).

Public API:
  - default_params() / AutoPauseParams      fixed trip & resume thresholds
  - REGISTRY / get_adapter()                per-strategy monitor_only handling
  - get_monitor_only() / set_monitor_only() flip a cell (strategy_config DB)
  - load_closed_trades()                    recent closed trades (forex DB)
  - evaluate_performance() / decide_trip()  PF + streak, fixed trip rule
  - load_eligible_cells()                   frozen-baseline allowlist
"""
from __future__ import annotations

from .config import (
    AutoPauseParams,
    default_params,
    forex_dsn,
    strategy_config_dsn,
)
from .adapters import (
    CONFIG_ID_BY_ENV,
    REGISTRY,
    StrategyAdapter,
    build_scope_clause,
    get_adapter,
    get_monitor_only,
    monitor_only_select_expr,
    monitor_only_set_sql,
    refresh_config_cache,
    set_monitor_only,
)
from .evaluator import (
    PerfStats,
    TripDecision,
    decide_trip,
    evaluate_performance,
    load_closed_trades,
)
from .eligibility import EligibilityRecord, load_eligible_cells
from .state import (
    PauseState,
    get_pause_state,
    record_eval,
    record_pause,
    record_resume,
)
from .shadow import (
    compute_shadow_stats,
    get_cell_sl_tp,
    is_long,
    pip_value,
    simulate_outcome,
)
from .resume import ResumeProposal, evaluate_resume

__all__ = [
    "AutoPauseParams",
    "default_params",
    "forex_dsn",
    "strategy_config_dsn",
    "CONFIG_ID_BY_ENV",
    "REGISTRY",
    "StrategyAdapter",
    "build_scope_clause",
    "get_adapter",
    "get_monitor_only",
    "monitor_only_select_expr",
    "monitor_only_set_sql",
    "refresh_config_cache",
    "set_monitor_only",
    "PerfStats",
    "TripDecision",
    "decide_trip",
    "evaluate_performance",
    "load_closed_trades",
    "EligibilityRecord",
    "load_eligible_cells",
    "PauseState",
    "get_pause_state",
    "record_eval",
    "record_pause",
    "record_resume",
    "compute_shadow_stats",
    "get_cell_sl_tp",
    "is_long",
    "pip_value",
    "simulate_outcome",
    "ResumeProposal",
    "evaluate_resume",
]
