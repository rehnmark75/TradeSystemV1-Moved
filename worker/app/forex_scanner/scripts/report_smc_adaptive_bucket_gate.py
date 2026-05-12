#!/usr/bin/env python3
"""Report live adaptive bucket gate evidence from alert_history.

This is the promotion guardrail: review MONITORING evidence before switching
adaptive_bucket_gate_mode to ACTIVE.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List

import psycopg2
import psycopg2.extras


DEFAULT_DSN = os.getenv(
    "TRADING_DSN",
    "host=postgres dbname=forex user=postgres password=postgres",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report SMC adaptive bucket gate evidence")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--epic", default="CS.D.EURUSD.CEEM.IP")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--environment", default=os.getenv("TRADING_ENVIRONMENT", "demo"))
    return parser.parse_args()


def print_rows(title: str, rows: List[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("no rows")
        return
    for row in rows:
        print(" | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> int:
    args = parse_args()
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    adaptive_bucket_gate_state AS state,
                    adaptive_bucket_gate_bucket AS bucket,
                    count(*) AS alerts,
                    count(*) FILTER (WHERE adaptive_bucket_gate_would_block) AS would_block,
                    round(avg(adaptive_bucket_gate_win_rate)::numeric, 3) AS avg_bucket_wr,
                    round(avg(adaptive_bucket_gate_trades)::numeric, 1) AS avg_bucket_trades,
                    max(alert_timestamp) AS last_alert
                FROM alert_history
                WHERE strategy = 'SMC_SIMPLE'
                  AND epic = %s
                  AND coalesce(environment, %s) = %s
                  AND alert_timestamp >= now() - (%s || ' days')::interval
                  AND adaptive_bucket_gate_state IS NOT NULL
                GROUP BY 1, 2
                ORDER BY would_block DESC, alerts DESC, bucket
                """,
                (args.epic, args.environment, args.environment, args.days),
            )
            print_rows("Adaptive Gate Buckets", [dict(row) for row in cur.fetchall()])

            cur.execute(
                """
                SELECT
                    alert_timestamp,
                    signal_type,
                    confidence_score,
                    adaptive_bucket_gate_state AS state,
                    adaptive_bucket_gate_bucket AS bucket,
                    adaptive_bucket_gate_trades AS trades,
                    adaptive_bucket_gate_win_rate AS win_rate,
                    adaptive_bucket_gate_reason AS reason
                FROM alert_history
                WHERE strategy = 'SMC_SIMPLE'
                  AND epic = %s
                  AND coalesce(environment, %s) = %s
                  AND alert_timestamp >= now() - (%s || ' days')::interval
                  AND adaptive_bucket_gate_would_block
                ORDER BY alert_timestamp DESC
                LIMIT 20
                """,
                (args.epic, args.environment, args.environment, args.days),
            )
            print_rows("Recent Would-Blocks", [dict(row) for row in cur.fetchall()])
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
