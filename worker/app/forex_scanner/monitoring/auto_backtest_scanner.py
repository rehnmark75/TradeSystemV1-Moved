#!/usr/bin/env python3
"""Auto-backtest scanner — closes the loop between the Monitor Outcomes board
and the bt.py --live-parity promotion gate.

Scans monitor_only_outcomes (forex DB) for (strategy, epic) cells whose
bracket-independent edge metrics clear the trigger gates (same metrics the
trading-ui monitor-outcomes page computes: edge_ratio = avg MFE / avg MAE,
% of signals with MFE > MAE, dead-on-arrival share), then runs a full
bt.py --live-parity backtest for each candidate and records the verdict in
auto_backtest_runs (strategy_config DB). system-monitor delivers verdicts to
Telegram (see system-monitor/app/services/auto_backtest_watcher.py).

The verdict is a TRIAGE result, not a promotion: PROMOTION_CANDIDATE means
"the live-parity gate that killed past mirages did not kill this one —
worth a human review". Nothing is ever auto-enabled.

Designed to run as its own container (auto-backtest in docker-compose) so the
heavy backtests never block the live scan loop:

    python3 /app/forex_scanner/monitoring/auto_backtest_scanner.py --loop

Modes:
    --once      one detection+run cycle, then exit
    --loop      run forever, sleeping --interval seconds between cycles
    --dry-run   detection only: print candidates, run nothing, write nothing
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("auto_backtest")

FOREX_DB_URL = os.getenv("DATABASE_URL",
                         "postgresql://postgres:postgres@postgres:5432/forex")
STRATEGY_CONFIG_DB_URL = os.getenv(
    "STRATEGY_CONFIG_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/strategy_config")

# --- Trigger gates (bracket-independent, mirrors monitor-outcomes UI) -------
WINDOW_DAYS = int(os.getenv("AUTO_BT_WINDOW_DAYS", "60"))
MIN_N = int(os.getenv("AUTO_BT_MIN_N", "30"))
MIN_EDGE_RATIO = float(os.getenv("AUTO_BT_MIN_EDGE_RATIO", "1.3"))
MIN_PCT_FAVORABLE = float(os.getenv("AUTO_BT_MIN_PCT_FAVORABLE", "55"))
MAX_DEAD_ON_ARRIVAL = float(os.getenv("AUTO_BT_MAX_DOA_PCT", "35"))
ENVIRONMENT = os.getenv("AUTO_BT_ENVIRONMENT", "demo")

# --- Run pacing --------------------------------------------------------------
BACKTEST_DAYS = int(os.getenv("AUTO_BT_DAYS", "90"))  # 30d regresses on 90d; gate on 90d
COOLDOWN_DAYS = int(os.getenv("AUTO_BT_COOLDOWN_DAYS", "21"))
FAILED_RETRY_DAYS = int(os.getenv("AUTO_BT_FAILED_RETRY_DAYS", "3"))
MAX_PER_CYCLE = int(os.getenv("AUTO_BT_MAX_PER_CYCLE", "1"))
SUBPROCESS_TIMEOUT = int(os.getenv("AUTO_BT_TIMEOUT_SECONDS", "14400"))  # 4h

# --- Verdict gates ------------------------------------------------------------
VERDICT_MIN_PF = float(os.getenv("AUTO_BT_VERDICT_MIN_PF", "1.3"))
VERDICT_MIN_TRADES = int(os.getenv("AUTO_BT_VERDICT_MIN_TRADES", "20"))

# Strategies whose live/demo behaviour requires extra bt.py flags for parity.
# SMC_SIMPLE regression must use --scalp (memory: feedback_smc_regression_scalp).
EXTRA_ARGS = {
    "SMC_SIMPLE": ["--scalp"],
    "SMC_SIMPLE_V2": ["--scalp"],
}

CANDIDATE_SQL = """
    SELECT strategy, epic, MAX(pair) AS pair,
           COUNT(*) AS n,
           ROUND(AVG(mfe_pips), 2) AS avg_mfe,
           ROUND(AVG(mae_pips), 2) AS avg_mae,
           ROUND(100.0 * SUM((mfe_pips > mae_pips)::int) / COUNT(*), 0)
               AS pct_mfe_favorable,
           ROUND(100.0 * SUM((mfe_pips < 2)::int) / COUNT(*), 1)
               AS dead_on_arrival_pct,
           CASE WHEN SUM(CASE WHEN ref_pnl_pips < 0 THEN -ref_pnl_pips END) > 0
                THEN ROUND(SUM(CASE WHEN ref_pnl_pips > 0 THEN ref_pnl_pips ELSE 0 END)
                     / SUM(CASE WHEN ref_pnl_pips < 0 THEN -ref_pnl_pips ELSE 0 END), 2)
           END AS ref_pf
    FROM monitor_only_outcomes
    WHERE status = 'RESOLVED'
      AND was_executed = FALSE
      AND environment = %s
      AND signal_timestamp > now() - (%s || ' days')::interval
      AND mfe_pips IS NOT NULL AND mae_pips IS NOT NULL
    GROUP BY strategy, epic
"""


def detect_candidates():
    """Return cells clearing the trigger gates, best edge first."""
    with psycopg2.connect(FOREX_DB_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(CANDIDATE_SQL, [ENVIRONMENT, str(WINDOW_DAYS)])
            cells = [dict(r) for r in cur.fetchall()]

    candidates = []
    for c in cells:
        avg_mae = float(c["avg_mae"] or 0)
        if avg_mae <= 0:
            continue
        edge_ratio = round(float(c["avg_mfe"]) / avg_mae, 2)
        n = int(c["n"])
        if (n >= MIN_N
                and edge_ratio >= MIN_EDGE_RATIO
                and float(c["pct_mfe_favorable"]) >= MIN_PCT_FAVORABLE
                and float(c["dead_on_arrival_pct"]) <= MAX_DEAD_ON_ARRIVAL):
            c["edge_ratio"] = edge_ratio
            c["per_month"] = round(n / WINDOW_DAYS * 30, 1)
            c["window_days"] = WINDOW_DAYS
            candidates.append(c)
    candidates.sort(key=lambda c: c["edge_ratio"], reverse=True)
    return candidates


def in_cooldown(conn, strategy, epic):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM auto_backtest_runs
            WHERE strategy = %s AND epic = %s AND environment = %s
              AND created_at > now() - (
                    CASE WHEN status = 'FAILED' THEN %s ELSE %s END
                    || ' days')::interval
            LIMIT 1
            """,
            [strategy, epic, ENVIRONMENT,
             str(FAILED_RETRY_DAYS), str(COOLDOWN_DAYS)],
        )
        return cur.fetchone() is not None


def build_command(strategy, epic):
    # Days go as a bare digit: bt.py rewrites N -> "--days N" itself, and an
    # explicit --days flag gets duplicated by its pass-through parser.
    cmd = ["python3", "-u", "/app/forex_scanner/bt.py",
           "--epic", epic,
           "--strategy", strategy,
           str(BACKTEST_DAYS),
           "--live-parity"]
    cmd += EXTRA_ARGS.get(strategy, [])
    return cmd


def _last_float(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def parse_backtest_output(output):
    """Parse the backtest_order_logger summary block (last occurrence wins)."""
    winners = _last_float(r"Winners:\s*(\d+)", output)
    losers = _last_float(r"Losers:\s*(\d+)", output)
    breakevens = _last_float(r"Breakeven:\s*(\d+)", output) or 0
    results = {
        "pf": _last_float(r"Profit Factor:\s*([\d.]+)", output),
        "win_rate_pct": _last_float(r"Win Rate:\s*([\d.]+)%", output),
        "expectancy_pips": _last_float(r"Expectancy:\s*(-?[\d.]+)\s*pips", output),
        "winners": int(winners) if winners is not None else None,
        "losers": int(losers) if losers is not None else None,
        "breakevens": int(breakevens),
    }
    closed = None
    if winners is not None or losers is not None:
        closed = int((winners or 0) + (losers or 0) + breakevens)
    results["total_closed"] = closed
    results["parse_ok"] = results["win_rate_pct"] is not None or closed is not None
    return results


def decide_verdict(results):
    closed = results.get("total_closed")
    pf = results.get("pf")
    exp = results.get("expectancy_pips")
    if not results.get("parse_ok"):
        return "UNKNOWN"
    if not closed:
        return "NO_SIGNALS"
    if pf is None and results.get("losers") == 0 and results.get("winners"):
        pf = 999.0  # no losing trades: PF line is not printed
        results["pf"] = pf
    if pf is None:
        return "UNKNOWN"
    if pf >= VERDICT_MIN_PF and closed >= VERDICT_MIN_TRADES and (exp or 0) > 0:
        return "PROMOTION_CANDIDATE"
    if pf >= 1.0 and closed >= 10:
        return "MARGINAL"
    return "NO_GO"


def run_backtest(cfg_conn, candidate):
    strategy, epic = candidate["strategy"], candidate["epic"]
    cmd = build_command(strategy, epic)
    trigger_metrics = {
        k: (float(candidate[k]) if candidate[k] is not None else None)
        for k in ("n", "edge_ratio", "pct_mfe_favorable", "dead_on_arrival_pct",
                  "avg_mfe", "avg_mae", "ref_pf", "per_month", "window_days")
    }
    with cfg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO auto_backtest_runs
                (strategy, epic, pair, environment, trigger_metrics,
                 backtest_days, command, status, started_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'RUNNING', now())
            RETURNING id
            """,
            [strategy, epic, candidate.get("pair"), ENVIRONMENT,
             json.dumps(trigger_metrics), BACKTEST_DAYS, " ".join(cmd)],
        )
        run_id = cur.fetchone()[0]
    cfg_conn.commit()

    logger.info(f"[run {run_id}] {strategy} {epic}: {' '.join(cmd)}")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=SUBPROCESS_TIMEOUT,
                              cwd="/app/forex_scanner")
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"bt.py exited {proc.returncode}; tail:\n{output[-4000:]}")
        results = parse_backtest_output(output)
        verdict = decide_verdict(results)
        error = None if results["parse_ok"] else f"parse failed; tail:\n{output[-4000:]}"
        status = "COMPLETED"
    except Exception as exc:  # timeout, crash, parse hard-failure
        results, verdict, status = None, None, "FAILED"
        error = str(exc)[:8000]

    elapsed = round(time.monotonic() - started)
    with cfg_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE auto_backtest_runs
            SET status = %s, results = %s, verdict = %s, error = %s,
                completed_at = now()
            WHERE id = %s
            """,
            [status, json.dumps(results) if results else None,
             verdict, error, run_id],
        )
    cfg_conn.commit()
    logger.info(f"[run {run_id}] {status} in {elapsed}s — verdict={verdict} "
                f"results={results}")


def run_cycle(dry_run=False):
    candidates = detect_candidates()
    logger.info(f"Detection: {len(candidates)} cell(s) clear the trigger gates "
                f"(window {WINDOW_DAYS}d, env {ENVIRONMENT})")
    for c in candidates:
        logger.info(f"  candidate: {c['strategy']} {c['epic']} "
                    f"edge={c['edge_ratio']} fav={c['pct_mfe_favorable']}% "
                    f"n={c['n']} doa={c['dead_on_arrival_pct']}% ref_pf={c['ref_pf']}")
    if dry_run or not candidates:
        return

    cfg_conn = psycopg2.connect(STRATEGY_CONFIG_DB_URL)
    try:
        launched = 0
        for c in candidates:
            if launched >= MAX_PER_CYCLE:
                logger.info(f"Cycle cap reached ({MAX_PER_CYCLE}); "
                            f"remaining candidates wait for the next cycle")
                break
            if in_cooldown(cfg_conn, c["strategy"], c["epic"]):
                logger.info(f"  skip (cooldown): {c['strategy']} {c['epic']}")
                continue
            run_backtest(cfg_conn, c)
            launched += 1
    finally:
        cfg_conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="one cycle, then exit")
    parser.add_argument("--loop", action="store_true", help="run forever")
    parser.add_argument("--interval", type=int, default=21600,
                        help="seconds between cycles in --loop mode (default 6h)")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect and print candidates only")
    args = parser.parse_args()

    if not (args.once or args.loop or args.dry_run):
        parser.error("choose one of --once / --loop / --dry-run")

    if args.dry_run:
        run_cycle(dry_run=True)
        return

    while True:
        try:
            run_cycle()
        except Exception:
            logger.exception("auto-backtest cycle failed")
        if not args.loop:
            break
        logger.info(f"Sleeping {args.interval}s until next cycle "
                    f"({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
