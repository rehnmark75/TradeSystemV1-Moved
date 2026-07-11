"""Replay Trip Rule B over a cell's historical shadow series.

Walks monitor_only_outcomes chronologically for a (strategy, epic, config_set)
cell: freezes the baseline from the FIRST --baseline-n resolved outcomes, then
slides the trip window over the rest, printing every trip and (hysteresis)
resume transition. This is the acceptance test for the thresholds:

  * must trip cells that really decayed (RANGE_FADE June collapse,
    SMC_SIMPLE ~May degradation)
  * must NOT trip healthy cells more than ~once over the full history
  * must not flip-flop trip/resume

Do NOT grid-search thresholds against this — that reintroduces single-regime
overfit. If a known decay is missed, the rule is wrong in kind (revisit window
size only).

Usage (inside task-worker):
    python /app/forex_scanner/scripts/auto_pause_replay.py \
        --strategy RANGE_FADE --epic CS.D.EURUSD.CEEM.IP [--config-set demo] \
        [--days 180] [--baseline-n 50] [--verbose]

    # sweep every cell with enough data
    python /app/forex_scanner/scripts/auto_pause_replay.py --all [--days 180]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional

import psycopg2
import psycopg2.extras

sys.path.insert(0, "/app")

from forex_scanner.core.trading.auto_pause import (  # noqa: E402
    default_params,
    decide_trip_shadow,
    evaluate_performance,
    forex_dsn,
)


@dataclass
class Transition:
    when: str
    kind: str      # 'TRIP' | 'RESUME'
    reason: str
    idx: int


def load_series(strategy: str, epic: str, config_set: str, days: int):
    query = """
        SELECT signal_timestamp, ref_pnl_pips
        FROM monitor_only_outcomes
        WHERE strategy = %s
          AND epic = %s
          AND coalesce(environment, %s) = %s
          AND status = 'RESOLVED'
          AND ref_pnl_pips IS NOT NULL
          AND signal_timestamp >= now() - make_interval(days => %s)
        ORDER BY signal_timestamp ASC
    """
    with psycopg2.connect(forex_dsn()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, [strategy, epic, config_set, config_set, days])
            return [dict(r) for r in cur.fetchall()]


def list_cells(config_set: str, days: int, min_n: int):
    query = """
        SELECT strategy, epic, count(*) AS n
        FROM monitor_only_outcomes
        WHERE coalesce(environment, %s) = %s
          AND status = 'RESOLVED'
          AND ref_pnl_pips IS NOT NULL
          AND signal_timestamp >= now() - make_interval(days => %s)
        GROUP BY strategy, epic
        HAVING count(*) >= %s
        ORDER BY strategy, epic
    """
    with psycopg2.connect(forex_dsn()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, [config_set, config_set, days, min_n])
            return [dict(r) for r in cur.fetchall()]


def replay_cell(strategy: str, epic: str, config_set: str, days: int,
                baseline_n: int, verbose: bool = False) -> Optional[dict]:
    params = default_params()
    series = load_series(strategy, epic, config_set, days)
    if len(series) < baseline_n + params.shadow_min_outcomes:
        return None

    baseline_rows = [
        {"profit_loss": r["ref_pnl_pips"]} for r in reversed(series[:baseline_n])
    ]
    baseline = evaluate_performance(baseline_rows)
    base_wr = baseline.win_rate

    transitions: List[Transition] = []
    paused = False
    for i in range(baseline_n, len(series) + 1):
        # trip window = most recent shadow_window outcomes at step i,
        # most-recent-first as evaluate_performance expects
        lo = max(0, i - params.shadow_window)
        window = [
            {"profit_loss": r["ref_pnl_pips"]} for r in reversed(series[lo:i])
        ]
        stats = evaluate_performance(window)
        ts = series[i - 1]["signal_timestamp"]
        when = f"{ts:%Y-%m-%d}"

        if not paused:
            decision = decide_trip_shadow(stats, base_wr, params)
            if decision.should_pause:
                paused = True
                transitions.append(Transition(when, "TRIP", decision.reason, i))
        else:
            # Resume hysteresis on the same series (PF and WR legs; the
            # cooldown/fresh-outcome gates are time-based and skipped here —
            # replay measures rule stability, not calendar pacing).
            pf_ok = stats.pf is not None and stats.pf > params.resume_pf_threshold
            wr_ok = stats.win_rate > base_wr - 0.05
            if pf_ok and wr_ok and stats.n >= params.resume_min_signals:
                paused = False
                transitions.append(Transition(
                    when, "RESUME",
                    f"PF {stats.pf:.2f} > {params.resume_pf_threshold}, "
                    f"WR {stats.win_rate:.0%}", i,
                ))

    trips = [t for t in transitions if t.kind == "TRIP"]
    pf_txt = "inf" if baseline.pf is None else f"{baseline.pf:.2f}"
    print(f"\n=== {strategy} {epic} ({config_set}) ===")
    print(f"  series: {len(series)} resolved outcomes "
          f"({series[0]['signal_timestamp']:%Y-%m-%d} .. "
          f"{series[-1]['signal_timestamp']:%Y-%m-%d})")
    print(f"  frozen baseline (first {baseline_n}): PF {pf_txt}, "
          f"WR {base_wr:.0%}")
    if not transitions:
        print("  no transitions — never tripped")
    for t in transitions:
        print(f"  {t.when}  {t.kind:6s} @outcome#{t.idx}  {t.reason}")
    if verbose:
        # end-state rolling stats
        tail = [{"profit_loss": r["ref_pnl_pips"]}
                for r in reversed(series[-params.shadow_window:])]
        st = evaluate_performance(tail)
        pf_now = "inf" if st.pf is None else f"{st.pf:.2f}"
        print(f"  current window: PF {pf_now}, WR {st.win_rate:.0%}, n={st.n}")
    return {
        "strategy": strategy, "epic": epic,
        "n": len(series), "baseline_pf": baseline.pf, "baseline_wr": base_wr,
        "trips": len(trips), "transitions": len(transitions),
        "first_trip": trips[0].when if trips else None,
        "ended_paused": paused,
    }


def main():
    ap = argparse.ArgumentParser(description="Replay auto-pause Trip Rule B on history")
    ap.add_argument("--strategy")
    ap.add_argument("--epic")
    ap.add_argument("--all", action="store_true", help="Replay every cell with enough data")
    ap.add_argument("--config-set", default="demo", choices=["demo", "live"])
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--baseline-n", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    params = default_params()
    min_needed = args.baseline_n + params.shadow_min_outcomes

    if args.all:
        cells = list_cells(args.config_set, args.days, min_needed)
        if not cells:
            print("No cells with enough shadow data.")
            return
        results = []
        for c in cells:
            r = replay_cell(c["strategy"], c["epic"], args.config_set,
                            args.days, args.baseline_n, args.verbose)
            if r:
                results.append(r)
        print("\n" + "=" * 72)
        print(f"{'strategy':18s} {'epic':24s} {'basePF':>7s} {'baseWR':>7s} "
              f"{'trips':>5s} {'first_trip':>11s} {'paused?':>8s}")
        for r in sorted(results, key=lambda x: (-x["trips"], x["strategy"])):
            pf_txt = " inf" if r["baseline_pf"] is None else f"{r['baseline_pf']:.2f}"
            print(f"{r['strategy']:18s} {r['epic']:24s} {pf_txt:>7s} "
                  f"{r['baseline_wr']:>6.0%} {r['trips']:>5d} "
                  f"{str(r['first_trip'] or '—'):>11s} "
                  f"{'yes' if r['ended_paused'] else 'no':>8s}")
        return

    if not args.strategy or not args.epic:
        ap.error("--strategy and --epic required (or --all)")
    r = replay_cell(args.strategy, args.epic, args.config_set,
                    args.days, args.baseline_n, args.verbose)
    if r is None:
        print(f"Not enough data (< {min_needed} resolved outcomes).")


if __name__ == "__main__":
    main()
