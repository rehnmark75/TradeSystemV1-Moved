"""
Backfill post-mortems for historical SMC_SIMPLE and RANGE_FADE trades.

Usage (inside fastapi-dev container):
    python backfill_postmortems.py [--dry-run] [--env demo|live] [--concurrency 3]

Targets: closed trades with P&L in the last 3 months for SMC_SIMPLE or RANGE_FADE
that do not yet have a trade_postmortem row.

Rate: batches of N concurrent calls, ~5s sleep between batches to stay well
within the Anthropic API rate limits and keep the system prompt cache warm.
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

STRATEGIES = ("SMC_SIMPLE", "RANGE_FADE")
LOOKBACK_MONTHS = 3


def get_eligible_trade_ids(db, environment: str) -> list[tuple[int, str]]:
    """Return (trade_id, environment) pairs that need a post-mortem."""
    from sqlalchemy import text

    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_MONTHS * 30)

    rows = db.execute(
        text("""
            SELECT tl.id, tl.environment
            FROM trade_log tl
            JOIN alert_history ah ON ah.id = tl.alert_id
            LEFT JOIN trade_postmortem pm
                ON pm.trade_id = tl.id AND pm.environment = tl.environment
            WHERE tl.status = 'closed'
              AND tl.profit_loss IS NOT NULL
              AND tl.timestamp >= :cutoff
              AND ah.strategy = ANY(:strategies)
              AND (:env = 'all' OR tl.environment = :env)
              AND pm.id IS NULL
            ORDER BY tl.timestamp ASC
        """),
        {
            "cutoff": cutoff,
            "strategies": list(STRATEGIES),
            "env": environment,
        },
    ).fetchall()

    return [(row[0], row[1]) for row in rows]


async def run_backfill(environment: str, concurrency: int, dry_run: bool) -> None:
    from services.db import SessionLocal
    from services.trade_postmortem_service import _run_pipeline

    db = SessionLocal()
    try:
        pairs = get_eligible_trade_ids(db, environment)
    finally:
        db.close()

    total = len(pairs)
    logger.info(f"Found {total} trades needing post-mortems (env={environment})")

    if total == 0:
        logger.info("Nothing to do.")
        return

    if dry_run:
        logger.info("[DRY RUN] Would process:")
        for trade_id, env in pairs[:20]:
            logger.info(f"  trade_id={trade_id} env={env}")
        if total > 20:
            logger.info(f"  ... and {total - 20} more")
        return

    done = 0
    failed = 0
    start = time.time()

    for i in range(0, total, concurrency):
        batch = pairs[i : i + concurrency]
        tasks = [_run_pipeline(trade_id, env) for trade_id, env in batch]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (trade_id, env), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"trade {trade_id}: {result}")
                failed += 1
            else:
                done += 1

        elapsed = time.time() - start
        rate = done / elapsed * 60 if elapsed > 0 else 0
        logger.info(
            f"Progress: {done + failed}/{total} "
            f"(done={done} failed={failed} rate={rate:.1f}/min)"
        )

        # Sleep between batches: keeps system prompt cache warm (< 5 min TTL)
        # and avoids hitting Anthropic rate limits
        if i + concurrency < total:
            await asyncio.sleep(5)

    elapsed = time.time() - start
    logger.info(
        f"Backfill complete: {done} succeeded, {failed} failed, "
        f"{elapsed:.0f}s elapsed"
    )

    # Rough cost estimate from actual observed token usage
    est_input = done * 940
    est_output = done * 411
    est_cost = (est_input / 1_000_000 * 3.00) + (est_output / 1_000_000 * 15.00)
    logger.info(
        f"Estimated cost: ~${est_cost:.2f} "
        f"({est_input:,} input + {est_output:,} output tokens)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill post-mortems for historical trades")
    parser.add_argument("--env", default="demo", choices=["demo", "live", "all"],
                        help="Environment filter (default: demo)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Concurrent API calls per batch (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List eligible trades without calling the API")
    args = parser.parse_args()

    asyncio.run(run_backfill(args.env, args.concurrency, args.dry_run))


if __name__ == "__main__":
    main()
