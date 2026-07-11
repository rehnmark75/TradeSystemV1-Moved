"""Enroll a (strategy, epic, config_set) cell into auto-pause with a FROZEN
shadow baseline (Trip Rule B).

Computes the enrollment-time baseline (PF / win-rate / n) from the cell's
historical ref-grid series in monitor_only_outcomes — by default the OLDEST
``--baseline-n`` resolved outcomes in the lookback window, so the baseline
reflects the established edge, not the possibly-already-decaying recent tail.
Manual values from a 90d backtest can be supplied instead.

Prints the proposed row; writes only with --commit.

Usage (inside task-worker):
    python /app/forex_scanner/scripts/auto_pause_enroll.py \
        --strategy RANGE_FADE --epic CS.D.USDCAD.MINI.IP [--config-set demo] \
        [--trip-source shadow] [--days 90] [--baseline-n 50] \
        [--manual-pf 1.3 --manual-wr 0.52 --manual-n 60 --source "90d bt"] \
        [--notes "..."] [--commit]

    # list current enrollment
    python /app/forex_scanner/scripts/auto_pause_enroll.py --list
"""
from __future__ import annotations

import argparse
import sys

import psycopg2
import psycopg2.extras

sys.path.insert(0, "/app")

from forex_scanner.core.trading.auto_pause import (  # noqa: E402
    evaluate_performance,
    forex_dsn,
    strategy_config_dsn,
)


def compute_shadow_baseline(strategy: str, epic: str, config_set: str,
                            days: int, baseline_n: int):
    """Oldest ``baseline_n`` resolved ref-grid outcomes in the window."""
    query = """
        SELECT ref_pnl_pips AS profit_loss,
               min(signal_timestamp) OVER () AS window_start,
               max(signal_timestamp) OVER () AS window_end
        FROM monitor_only_outcomes
        WHERE strategy = %s
          AND epic = %s
          AND coalesce(environment, %s) = %s
          AND status = 'RESOLVED'
          AND ref_pnl_pips IS NOT NULL
          AND signal_timestamp >= now() - make_interval(days => %s)
        ORDER BY signal_timestamp ASC
        LIMIT %s
    """
    with psycopg2.connect(forex_dsn()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, [strategy, epic, config_set, config_set,
                                days, baseline_n])
            rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return None, None
    stats = evaluate_performance(rows)
    span = f"{rows[0]['window_start']:%Y-%m-%d}..{rows[0]['window_end']:%Y-%m-%d}"
    source = (f"monitor_only_outcomes oldest {stats.n} of last {days}d "
              f"(window {span})")
    return stats, source


def upsert_eligibility(args, pf, wr, n, source):
    query = """
        INSERT INTO auto_pause_eligibility
            (strategy, epic, config_set, eligible, trip_source,
             baseline_shadow_pf, baseline_shadow_wr, baseline_shadow_n,
             baseline_source, notes, updated_at)
        VALUES (%s, %s, %s, TRUE, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (strategy, epic, config_set) DO UPDATE SET
            eligible = TRUE,
            trip_source = EXCLUDED.trip_source,
            baseline_shadow_pf = EXCLUDED.baseline_shadow_pf,
            baseline_shadow_wr = EXCLUDED.baseline_shadow_wr,
            baseline_shadow_n = EXCLUDED.baseline_shadow_n,
            baseline_source = EXCLUDED.baseline_source,
            notes = coalesce(EXCLUDED.notes, auto_pause_eligibility.notes),
            updated_at = now()
    """
    with psycopg2.connect(strategy_config_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [args.strategy, args.epic, args.config_set,
                                args.trip_source, pf, wr, n, source, args.notes])
        conn.commit()


def list_enrollment():
    query = """
        SELECT strategy, epic, config_set, eligible, trip_source,
               baseline_pf, baseline_shadow_pf, baseline_shadow_wr,
               baseline_shadow_n, auto_resume, baseline_source
        FROM auto_pause_eligibility
        ORDER BY strategy, epic, config_set
    """
    with psycopg2.connect(strategy_config_dsn()) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    if not rows:
        print("No enrollment rows.")
        return
    for r in rows:
        flag = "ELIGIBLE" if r["eligible"] else "disabled"
        print(f"{flag:8s} {r['strategy']:16s} {r['epic']:24s} {r['config_set']:5s} "
              f"src={r['trip_source']:6s} "
              f"shadow_pf={r['baseline_shadow_pf']} shadow_wr={r['baseline_shadow_wr']} "
              f"n={r['baseline_shadow_n']} auto_resume={r['auto_resume']}")


def main():
    ap = argparse.ArgumentParser(description="Enroll a cell in auto-pause (Trip Rule B)")
    ap.add_argument("--list", action="store_true", help="List enrollment and exit")
    ap.add_argument("--strategy")
    ap.add_argument("--epic")
    ap.add_argument("--config-set", default="demo", choices=["demo", "live"])
    ap.add_argument("--trip-source", default="shadow",
                    choices=["trades", "shadow", "both"])
    ap.add_argument("--days", type=int, default=90,
                    help="Lookback for the shadow baseline window")
    ap.add_argument("--baseline-n", type=int, default=50,
                    help="Baseline sample size (oldest N in window)")
    ap.add_argument("--manual-pf", type=float, help="Manual baseline PF (skips compute)")
    ap.add_argument("--manual-wr", type=float, help="Manual baseline WR (0-1)")
    ap.add_argument("--manual-n", type=int, help="Manual baseline n")
    ap.add_argument("--source", help="Provenance note for manual baselines")
    ap.add_argument("--notes")
    ap.add_argument("--commit", action="store_true", help="Write the row (default: print only)")
    args = ap.parse_args()

    if args.list:
        list_enrollment()
        return

    if not args.strategy or not args.epic:
        ap.error("--strategy and --epic are required (or use --list)")

    manual = args.manual_pf is not None or args.manual_wr is not None
    if manual:
        if args.manual_wr is None or args.manual_n is None:
            ap.error("manual baseline needs --manual-pf, --manual-wr and --manual-n")
        pf, wr, n = args.manual_pf, args.manual_wr, args.manual_n
        source = args.source or "manual"
    else:
        stats, source = compute_shadow_baseline(
            args.strategy, args.epic, args.config_set, args.days, args.baseline_n
        )
        if stats is None or stats.n == 0:
            print(f"❌ No resolved shadow outcomes for {args.strategy} {args.epic} "
                  f"({args.config_set}) in last {args.days}d — cannot enroll.")
            sys.exit(1)
        pf, wr, n = stats.pf, round(stats.win_rate, 4), stats.n
        if n < args.baseline_n:
            print(f"⚠️ Only {n} outcomes available (< requested {args.baseline_n}); "
                  f"baseline will be thin.")

    pf_txt = "inf (no losses)" if pf is None else f"{pf:.3f}"
    print(f"\n{args.strategy} {args.epic} ({args.config_set}) trip_source={args.trip_source}")
    print(f"  baseline_shadow_pf = {pf_txt}")
    print(f"  baseline_shadow_wr = {wr}")
    print(f"  baseline_shadow_n  = {n}")
    print(f"  baseline_source    = {source}")
    if wr is not None and wr < 0.40:
        print("  ⚠️ baseline WR below ref-grid breakeven (40% at RR 1.5) — this cell "
              "has no shadow edge to protect; enrolling it will trip ~constantly.")

    if not args.commit:
        print("\nDry run — re-run with --commit to write.")
        return
    upsert_eligibility(args, pf, wr, n, source)
    print("✅ Enrolled.")


if __name__ == "__main__":
    main()
