#!/usr/bin/env python3
"""Unit tests for the Auto-Pause layer Phase 1 building blocks.

Covers the pure (DB-free) functions: per-strategy scope-clause / monitor_only
SQL builders, the PF + consecutive-loss evaluator, and the fixed trip rule.

Run directly (no pytest needed):
    docker exec task-worker python /app/forex_scanner/tests/test_auto_pause.py
or via pytest:
    docker exec task-worker python -m pytest /app/forex_scanner/tests/test_auto_pause.py
"""
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta  # noqa: E402

from forex_scanner.core.trading.auto_pause import (  # noqa: E402
    AutoPauseParams,
    PerfStats,
    build_scope_clause,
    decide_trip,
    default_params,
    evaluate_performance,
    evaluate_resume,
    get_adapter,
    is_long,
    monitor_only_select_expr,
    monitor_only_set_sql,
    pip_value,
    simulate_outcome,
)

PARAMS = AutoPauseParams()  # fixed defaults


def r(value):
    """Build a trade row with profit_loss only (pips_gained NULL, as in prod)."""
    return {"pips_gained": None, "profit_loss": value}


# --------------------------------------------------------------------------- #
# Adapter registry + scope clauses
# --------------------------------------------------------------------------- #
def test_get_adapter_case_insensitive_and_unknown():
    assert get_adapter("range_fade").table == "range_fade_pair_overrides"
    assert get_adapter("RANGE_FADE").strategy == "RANGE_FADE"
    assert get_adapter("does_not_exist") is None


def test_scope_clause_config_id():
    # SMC_SIMPLE / IMPULSE_FADE use config_id (demo=3, live=2)
    a = get_adapter("SMC_SIMPLE")
    sql, params = build_scope_clause(a, "CS.D.EURUSD.CEEM.IP", "demo")
    assert sql == "epic = %s AND config_id = %s"
    assert params == ["CS.D.EURUSD.CEEM.IP", 3]
    _, live_params = build_scope_clause(a, "CS.D.EURUSD.CEEM.IP", "live")
    assert live_params == ["CS.D.EURUSD.CEEM.IP", 2]


def test_scope_clause_config_set():
    a = get_adapter("SMC_MOMENTUM")
    sql, params = build_scope_clause(a, "CS.D.EURJPY.MINI.IP", "demo")
    assert sql == "epic = %s AND config_set = %s"
    assert params == ["CS.D.EURJPY.MINI.IP", "demo"]


def test_scope_clause_none():
    a = get_adapter("DONCHIAN_TURTLE")
    sql, params = build_scope_clause(a, "CS.D.GBPUSD.MINI.IP", "demo")
    assert sql == "epic = %s"
    assert params == ["CS.D.GBPUSD.MINI.IP"]


def test_scope_clause_range_fade_profile():
    a = get_adapter("RANGE_FADE")
    sql, params = build_scope_clause(a, "CS.D.EURUSD.CEEM.IP", "demo")
    assert sql == "epic = %s AND config_set = %s AND profile_name = %s"
    assert params == ["CS.D.EURUSD.CEEM.IP", "demo", "5m"]


# --------------------------------------------------------------------------- #
# monitor_only SQL builders
# --------------------------------------------------------------------------- #
def test_set_sql_jsonb_pause_resume():
    a = get_adapter("SMC_SIMPLE")
    pause_sql, pause_params = monitor_only_set_sql(a, True)
    assert '"monitor_only": "true"' in pause_sql and "||" in pause_sql
    assert pause_params == []
    resume_sql, resume_params = monitor_only_set_sql(a, False)
    assert "- 'monitor_only'" in resume_sql
    assert resume_params == []


def test_set_sql_boolean_column():
    a = get_adapter("RANGE_FADE")
    sql, params = monitor_only_set_sql(a, True)
    assert sql == "monitor_only = %s"
    assert params == [True]


def test_select_expr():
    assert monitor_only_select_expr(get_adapter("SMC_SIMPLE")) == \
        "(parameter_overrides->>'monitor_only')::boolean"
    assert monitor_only_select_expr(get_adapter("RANGE_FADE")) == "monitor_only"


# --------------------------------------------------------------------------- #
# Performance evaluation
# --------------------------------------------------------------------------- #
def test_evaluate_pf_and_winrate():
    stats = evaluate_performance([r(10), r(-5)])
    assert stats.n == 2
    assert stats.gross_win == 10 and stats.gross_loss == 5
    assert abs(stats.pf - 2.0) < 1e-9
    assert abs(stats.win_rate - 0.5) < 1e-9


def test_evaluate_consecutive_losses_most_recent_first():
    # most-recent-first: -5, -3, (win 2 breaks), -1
    stats = evaluate_performance([r(-5), r(-3), r(2), r(-1)])
    assert stats.consecutive_losses == 2
    assert stats.n == 4
    assert abs(stats.pf - (2.0 / 9.0)) < 1e-9


def test_evaluate_breakeven_breaks_streak():
    assert evaluate_performance([r(0), r(-5)]).consecutive_losses == 0
    assert evaluate_performance([r(-5), r(0), r(-3)]).consecutive_losses == 1


def test_evaluate_no_losses_pf_none():
    stats = evaluate_performance([r(5), r(3)])
    assert stats.pf is None  # PF undefined with zero losses
    assert stats.consecutive_losses == 0


def test_evaluate_empty():
    stats = evaluate_performance([])
    assert stats.n == 0 and stats.pf is None and stats.consecutive_losses == 0


def test_evaluate_single_unit_skips_mixed_rows():
    # profit_loss present somewhere -> field=profit_loss; the pips-only row is
    # skipped so units are never mixed into PF.
    rows = [
        {"pips_gained": None, "profit_loss": 10},   # used
        {"pips_gained": 5, "profit_loss": None},    # skipped (no profit_loss)
        {"pips_gained": None, "profit_loss": -5},   # used
    ]
    stats = evaluate_performance(rows)
    assert stats.n == 2
    assert stats.gross_win == 10 and stats.gross_loss == 5
    assert abs(stats.pf - 2.0) < 1e-9


def test_evaluate_uses_pips_when_no_profit_loss():
    rows = [{"pips_gained": 8, "profit_loss": None},
            {"pips_gained": -4, "profit_loss": None}]
    stats = evaluate_performance(rows)
    assert stats.n == 2
    assert abs(stats.pf - 2.0) < 1e-9


# --------------------------------------------------------------------------- #
# Trip decision
# --------------------------------------------------------------------------- #
def test_trip_pf_trigger():
    stats = PerfStats(n=12, pf=0.5, gross_win=10, gross_loss=20, win_rate=0.4,
                      consecutive_losses=1)
    assert decide_trip(stats, PARAMS).should_pause is True


def test_trip_min_trades_gate_blocks_pf_trigger():
    # PF below threshold but too few trades and streak below limit -> no pause
    stats = PerfStats(n=8, pf=0.5, gross_win=10, gross_loss=20, win_rate=0.4,
                      consecutive_losses=2)
    assert decide_trip(stats, PARAMS).should_pause is False


def test_trip_consecutive_loss_trigger_fires_at_low_n():
    # 5 straight losses, only 5 trades -> consec trigger still fires
    stats = PerfStats(n=5, pf=0.0, gross_win=0.0, gross_loss=25, win_rate=0.0,
                      consecutive_losses=5)
    d = decide_trip(stats, PARAMS)
    assert d.should_pause is True and "consecutive" in d.reason


def test_trip_healthy_no_pause():
    stats = PerfStats(n=20, pf=1.5, gross_win=30, gross_loss=20, win_rate=0.6,
                      consecutive_losses=0)
    assert decide_trip(stats, PARAMS).should_pause is False


def test_trip_pf_none_no_pause():
    stats = PerfStats(n=10, pf=None, gross_win=30, gross_loss=0, win_rate=1.0,
                      consecutive_losses=0)
    assert decide_trip(stats, PARAMS).should_pause is False


def test_default_params_defaults():
    # default_params reads env; with nothing set it equals the fixed defaults
    p = default_params()
    assert p.trip_pf_threshold == 0.8
    assert p.trip_min_trades == 10
    assert p.trip_max_consecutive_losses == 5
    assert p.resume_pf_threshold == 1.1


# --------------------------------------------------------------------------- #
# Phase 3: shadow reconstruction (pure simulation) + resume rule
# --------------------------------------------------------------------------- #
def test_pip_value():
    assert pip_value("CS.D.EURUSD.CEEM.IP") == 0.0001
    assert pip_value("CS.D.USDJPY.MINI.IP") == 0.01
    assert pip_value("CS.D.CFEGOLD.CEE.IP") == 0.1


def test_is_long():
    assert is_long("BULL") is True and is_long("BUY") is True
    assert is_long("BEAR") is False and is_long("SELL") is False
    assert is_long("weird") is None


def _c(high, low):
    return {"high": high, "low": low}


def test_simulate_long_tp_hit():
    # entry 1.1000, tp +15 pips = 1.10150, sl -8 pips = 1.09920
    out = simulate_outcome(1.1000, True, 8, 15, 0.0001,
                           [_c(1.1010, 1.0998), _c(1.1016, 1.1005)])
    assert out == 15


def test_simulate_long_sl_hit():
    out = simulate_outcome(1.1000, True, 8, 15, 0.0001, [_c(1.1005, 1.0991)])
    assert out == -8


def test_simulate_ambiguous_resolves_to_sl():
    # one candle spans both levels -> conservative stop
    out = simulate_outcome(1.1000, True, 8, 15, 0.0001, [_c(1.1020, 1.0990)])
    assert out == -8


def test_simulate_unresolved_returns_none():
    out = simulate_outcome(1.1000, True, 8, 15, 0.0001, [_c(1.1005, 1.0996)])
    assert out is None


def test_simulate_short_tp_hit():
    # short entry 1.1000, tp = entry - 15 pips = 1.09850
    out = simulate_outcome(1.1000, False, 8, 15, 0.0001, [_c(1.1003, 1.0984)])
    assert out == 15


def _stats(pf, n=20):
    return PerfStats(n=n, pf=pf, gross_win=0.0, gross_loss=0.0,
                     win_rate=0.6, consecutive_losses=0)


def test_resume_proposes_when_recovered():
    paused = datetime.now() - timedelta(days=11)
    p = evaluate_resume(paused, _stats(1.3), 20, AutoPauseParams(), datetime.now())
    assert p.should_propose is True


def test_resume_blocks_insufficient_outcomes():
    paused = datetime.now() - timedelta(days=11)
    p = evaluate_resume(paused, _stats(1.3), 10, AutoPauseParams(), datetime.now())
    assert p.should_propose is False and "insufficient" in p.reason


def test_resume_blocks_cooldown():
    paused = datetime.now() - timedelta(days=5)
    p = evaluate_resume(paused, _stats(1.3), 20, AutoPauseParams(), datetime.now())
    assert p.should_propose is False and "cooldown" in p.reason


def test_resume_blocks_low_pf():
    paused = datetime.now() - timedelta(days=11)
    p = evaluate_resume(paused, _stats(1.0), 20, AutoPauseParams(), datetime.now())
    assert p.should_propose is False


def test_resume_blocks_pf_none():
    paused = datetime.now() - timedelta(days=11)
    p = evaluate_resume(paused, _stats(None), 20, AutoPauseParams(), datetime.now())
    assert p.should_propose is False


# --------------------------------------------------------------------------- #
# Minimal runner (so the file works without pytest installed)
# --------------------------------------------------------------------------- #
def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # pragma: no cover
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
