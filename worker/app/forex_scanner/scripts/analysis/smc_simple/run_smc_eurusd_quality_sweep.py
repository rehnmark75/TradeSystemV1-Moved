#!/usr/bin/env python3
"""Run a bounded EURUSD SMC_SIMPLE quality sweep.

This intentionally tests a small set of defensible strategy-quality changes
instead of broad parameter search. Each run is summarized from backtest_signals,
then replayed through the adaptive bucket gate evaluator.

Run inside task-worker:

    docker exec -i task-worker python /app/forex_scanner/scripts/analysis/smc_simple/run_smc_eurusd_quality_sweep.py --days 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)
DEFAULT_EPIC = "CS.D.EURUSD.CEEM.IP"
BACKTEST_CLI = Path("/app/forex_scanner/backtest_cli.py")
ADAPTIVE_EVAL = Path("/app/forex_scanner/scripts/analysis/smc_simple/eval_smc_adaptive_bucket_gate.py")


@dataclass
class SweepCase:
    name: str
    args: List[str] = field(default_factory=list)


CASES = [
    SweepCase("baseline"),
    SweepCase(
        "macd_no_continuation",
        [
            "--override", "macd_alignment_filter_enabled=true",
            "--pair-override", "EURUSD:macd_filter_enabled=true",
            "--pair-override", "EURUSD:continuation_entry_enabled=false",
        ],
    ),
    SweepCase(
        "macd_qualification_active",
        [
            "--override", "macd_alignment_filter_enabled=true",
            "--pair-override", "EURUSD:macd_filter_enabled=true",
            "--pair-override", "EURUSD:continuation_entry_enabled=false",
            "--qualification-active",
            "--qual-min-score", "0.55",
        ],
    ),
    SweepCase(
        "macd_active_micro_filters",
        [
            "--override", "macd_alignment_filter_enabled=true",
            "--pair-override", "EURUSD:macd_filter_enabled=true",
            "--pair-override", "EURUSD:continuation_entry_enabled=false",
            "--qualification-active",
            "--qual-min-score", "0.55",
            "--micro-all-filters",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded EURUSD SMC_SIMPLE sweep")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--epic", default=DEFAULT_EPIC)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--case", action="append", help="Run only matching case name; can be repeated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    return parser.parse_args()


def selected_cases(names: Optional[List[str]]) -> List[SweepCase]:
    if not names:
        return CASES
    wanted = set(names)
    cases = [case for case in CASES if case.name in wanted]
    missing = wanted - {case.name for case in cases}
    if missing:
        raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")
    return cases


def run_case(case: SweepCase, args: argparse.Namespace) -> Optional[int]:
    command = [
        "python",
        str(BACKTEST_CLI),
        "--epic",
        args.epic,
        "--days",
        str(args.days),
        "--strategy",
        "SMC_SIMPLE",
        "--timeframe",
        args.timeframe,
        *case.args,
    ]
    print(f"\n=== {case.name} ===", flush=True)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w+", prefix=f"smc_{case.name}_", suffix=".log", delete=not args.keep_logs
    ) as log_file:
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.seek(0)
        text = log_file.read()

        if args.keep_logs:
            print(f"log={log_file.name}", flush=True)
        if proc.returncode != 0:
            print(f"case failed: returncode={proc.returncode}", flush=True)
            print(text[-3000:], flush=True)
            return None

    matches = re.findall(r"Backtest execution (\d+) finalized", text)
    if not matches:
        matches = re.findall(r"execution[ _-]?id[=: ]+(\d+)", text, flags=re.IGNORECASE)
    if not matches:
        print("could not parse execution id", flush=True)
        print(text[-3000:], flush=True)
        return None
    execution_id = int(matches[-1])
    print(f"execution_id={execution_id}", flush=True)
    return execution_id


def summarize_execution(conn, execution_id: int, epic: str) -> Dict[str, Any]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                count(*) AS n,
                count(*) FILTER (WHERE trade_result = 'win') AS wins,
                count(*) FILTER (WHERE trade_result = 'loss') AS losses,
                round(sum(pips_gained)::numeric, 2) AS total_pips,
                round(avg(pips_gained)::numeric, 2) AS avg_pips,
                round(
                    coalesce(sum(pips_gained) FILTER (WHERE pips_gained > 0), 0)
                    / nullif(abs(coalesce(sum(pips_gained) FILTER (WHERE pips_gained < 0), 0)), 0),
                    2
                ) AS profit_factor
            FROM backtest_signals
            WHERE execution_id = %s
              AND epic = %s
              AND trade_result IN ('win', 'loss')
            """,
            (execution_id, epic),
        )
        summary = dict(cur.fetchone())

        cur.execute(
            """
            SELECT
                signal_type,
                count(*) AS n,
                count(*) FILTER (WHERE trade_result = 'win') AS wins,
                round(sum(pips_gained)::numeric, 2) AS total_pips,
                round(avg(pips_gained)::numeric, 2) AS avg_pips
            FROM backtest_signals
            WHERE execution_id = %s
              AND epic = %s
              AND trade_result IN ('win', 'loss')
            GROUP BY signal_type
            ORDER BY signal_type
            """,
            (execution_id, epic),
        )
        summary["by_direction"] = [dict(row) for row in cur.fetchall()]
    return summary


def run_adaptive_eval(execution_id: int, epic: str) -> str:
    command = [
        "python",
        str(ADAPTIVE_EVAL),
        "--execution-id",
        str(execution_id),
        "--epic",
        epic,
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    lines = proc.stdout.splitlines()
    return "\n".join(lines[:8])


def main() -> int:
    args = parse_args()
    cases = selected_cases(args.case)
    if args.dry_run:
        for case in cases:
            run_case(case, args)
        return 0

    conn = psycopg2.connect(args.dsn)
    try:
        rows = []
        for case in cases:
            execution_id = run_case(case, args)
            if execution_id is None:
                continue
            summary = summarize_execution(conn, execution_id, args.epic)
            summary["case"] = case.name
            summary["execution_id"] = execution_id
            rows.append(summary)
            print(json.dumps(summary, default=str, indent=2), flush=True)
            print(run_adaptive_eval(execution_id, args.epic), flush=True)
    finally:
        conn.close()

    print("\n=== ranked by EURUSD total_pips ===", flush=True)
    for row in sorted(rows, key=lambda item: float(item["total_pips"] or 0), reverse=True):
        print(
            f"{row['case']:28s} exec={row['execution_id']} "
            f"n={row['n']:4d} wins={row['wins']:4d} losses={row['losses']:4d} "
            f"pf={row['profit_factor']} pips={row['total_pips']} avg={row['avg_pips']}"
            , flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
