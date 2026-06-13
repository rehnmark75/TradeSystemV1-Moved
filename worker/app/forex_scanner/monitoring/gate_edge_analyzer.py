"""
Gate Edge Analyzer

Per-GATE counterfactual edge scoring for setups a non-SMC strategy REJECTED.

Companion to monitor_outcome_analyzer.py (which scores LOGGED-but-not-executed
signals). This one scores the upstream population: setups REJECTED at an edge
gate (ADX ceiling, ER floor, confidence floor, cooldown, …). Each gate gets a
measured PF for the population it blocked — without changing any live behaviour.

NOTE ON NAMING: distinct from the SMC-only `rejection_outcome_analyzer.py`
(class RejectionOutcomeAnalyzer), which reads smc_simple_rejections →
smc_rejection_outcomes. This module covers the non-SMC strategies whose
rejections flow through StrategyRejectionManager into strategy_rejections
(RANGE_FADE, MEAN_REVERSION, IMPULSE_FADE, XAU_GOLD).

Why it exists
-------------
RANGE_FADE (and siblings) stack ~20 sequential reject points. Several were added
to chase edge but they also suppress the monitor-phase data needed to decide
whether an epic should go active. This layer turns every gate from an invisible
volume sink into a measurable lever (read v_rejection_gate_edge):

    blocked_pf < ~1.0  → gate blocks net losers   → KEEP
    blocked_pf > ~1.3  → gate blocks net winners  → REVIEW (costing edge + data)

Population
----------
Source = `strategy_rejections` (strategy_config DB) rows with direction IS NOT
NULL. By construction in the strategies, a rejection only carries a direction
once a real directional trigger has been established (band touch + RSI extreme +
range proximity + HTF bias). So direction-bearing rejections are exactly the
"real setup, blocked by an edge gate" cases; structural rejections (no_trigger,
insufficient_data, session_blocked, …) carry direction NULL and are skipped.

Simulation
----------
Reuses MonitorOutcomeAnalyzer.simulate() verbatim. Rejections store no price, so
the entry is reconstructed from the first post-signal 1m candle open (the same
next-bar fill the monitor analyzer uses for price=0 rows), then MFE/MAE and a
fixed reference SL/TP grid are walked forward over a 24h horizon from ig_candles.
Results upsert into `rejection_outcomes` (forex DB); OPEN/NO_DATA rows are
re-evaluated on later runs until the 24h window completes.

CAVEAT: consecutive 5m scans on a persistent setup produce repeated rejections
at the same gate (especially `cooldown`), so per-gate n is inflated vs distinct
setups. PF/win-rate per gate stays directionally meaningful; treat the cooldown
gate's n with extra caution.

Usage:
    docker exec -it task-worker python /app/forex_scanner/monitoring/gate_edge_analyzer.py
    docker exec -it task-worker python /app/forex_scanner/monitoring/gate_edge_analyzer.py --days 30 --strategy RANGE_FADE
    docker exec -it task-worker python /app/forex_scanner/monitoring/gate_edge_analyzer.py --dry-run

Invoked automatically (once/6h) by the live scanner via
TradingOrchestrator._run_gate_edge_analysis_if_due().

Created: 2026-06-13
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from monitoring.monitor_outcome_analyzer import (
        MonitorOutcomeAnalyzer,
        MIN_SIGNAL_AGE_MINUTES,
    )
except ImportError:
    from forex_scanner.monitoring.monitor_outcome_analyzer import (
        MonitorOutcomeAnalyzer,
        MIN_SIGNAL_AGE_MINUTES,
    )

logger = logging.getLogger(__name__)

STRATEGY_CONFIG_URL = os.getenv(
    "STRATEGY_CONFIG_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/strategy_config",
)

# Default scope: the strategy whose data starvation prompted this layer. Any
# strategy that writes to strategy_rejections can be passed via --strategy.
DEFAULT_STRATEGY = "RANGE_FADE"


class GateEdgeAnalyzer(MonitorOutcomeAnalyzer):
    """Score the forward edge of GATED (rejected) setups, per gate.

    Inherits simulate()/fetch_candles()/ref-grid helpers from
    MonitorOutcomeAnalyzer (forex DB) and adds a read-only connection to
    strategy_config for the rejection source rows.
    """

    def __init__(self):
        super().__init__()
        self.strategy_config = create_engine(STRATEGY_CONFIG_URL)

    # -- rejection selection (strategy_config DB) ---------------------------

    def get_candidate_rejections(self, start_date, end_date, strategy: str) -> pd.DataFrame:
        """Direction-bearing rejections old enough to have a forward candle.

        Cross-DB re-evaluation: strategy_rejections lives in strategy_config and
        rejection_outcomes in forex, so we can't LEFT JOIN. Instead we exclude
        rejection_ids already RESOLVED in forex and let the idempotent upsert
        re-evaluate OPEN/NO_DATA rows.
        """
        max_signal_time = end_date - timedelta(minutes=MIN_SIGNAL_AGE_MINUTES)
        resolved_ids = self._resolved_rejection_ids(start_date)

        q = """
            SELECT id AS rejection_id, strategy, epic, pair, scan_timestamp,
                   stage, reason, direction, hour_utc, session
            FROM strategy_rejections
            WHERE strategy = :strategy
              AND direction IS NOT NULL
              AND scan_timestamp >= :start_date
              AND scan_timestamp <  :end_date
              AND scan_timestamp <= :max_signal_time
            ORDER BY scan_timestamp ASC
        """
        with self.strategy_config.connect() as conn:
            df = pd.DataFrame(
                conn.execute(
                    text(q),
                    {
                        "strategy": strategy,
                        "start_date": start_date,
                        "end_date": end_date,
                        "max_signal_time": max_signal_time,
                    },
                ).mappings().all()
            )
        if not df.empty and resolved_ids:
            df = df[~df["rejection_id"].isin(list(resolved_ids))]
        return df

    def _resolved_rejection_ids(self, start_date) -> set:
        """rejection_ids already finalized (RESOLVED) in forex.rejection_outcomes."""
        q = """
            SELECT rejection_id FROM rejection_outcomes
            WHERE status = 'RESOLVED' AND signal_timestamp >= :start_date
        """
        with self.forex.connect() as conn:
            rows = conn.execute(text(q), {"start_date": start_date}).mappings().all()
        return {r["rejection_id"] for r in rows}

    # -- adapt a rejection row into the sig dict simulate() expects ---------

    @staticmethod
    def _rejection_to_sig(rej: dict) -> dict:
        # price=0 forces simulate() to reconstruct the entry from the first
        # post-signal candle open (rejections store no price). signal_type drives
        # direction; alert_id carries the rejection id through simulate()'s output.
        # strategy_rejections.scan_timestamp is TIMESTAMPTZ; simulate() works in
        # naive-UTC (like alert_history + ig_candles), so normalize to naive UTC.
        ts = pd.to_datetime(rej["scan_timestamp"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return {
            "alert_id": int(rej["rejection_id"]),
            "strategy": rej["strategy"],
            "epic": rej["epic"],
            "pair": rej.get("pair"),
            "environment": None,
            "signal_type": rej["direction"],
            "alert_timestamp": ts.to_pydatetime(),
            "price": 0.0,
            "bid_price": None,
            "ask_price": None,
        }

    # -- persistence (forex DB) --------------------------------------------

    UPSERT = text("""
        INSERT INTO rejection_outcomes (
            rejection_id, strategy, epic, pair, stage, reason, session, hour_utc,
            signal_timestamp, direction, entry_price, pip_multiplier, horizon_minutes,
            mfe_pips, mae_pips, early_mae_pips, time_to_mfe_minutes, time_to_mae_minutes,
            pnl_60m_pips, pnl_240m_pips, pnl_1440m_pips,
            ref_sl_pips, ref_tp_pips, ref_outcome, ref_pnl_pips,
            time_to_tp_minutes, time_to_sl_minutes,
            candles_evaluated, status, evaluated_until, updated_at
        ) VALUES (
            :rejection_id, :strategy, :epic, :pair, :stage, :reason, :session, :hour_utc,
            :signal_timestamp, :direction, :entry_price, :pip_multiplier, :horizon_minutes,
            :mfe_pips, :mae_pips, :early_mae_pips, :time_to_mfe_minutes, :time_to_mae_minutes,
            :pnl_60m_pips, :pnl_240m_pips, :pnl_1440m_pips,
            :ref_sl_pips, :ref_tp_pips, :ref_outcome, :ref_pnl_pips,
            :time_to_tp_minutes, :time_to_sl_minutes,
            :candles_evaluated, :status, :evaluated_until, now()
        )
        ON CONFLICT (rejection_id) DO UPDATE SET
            entry_price = EXCLUDED.entry_price,
            mfe_pips = EXCLUDED.mfe_pips, mae_pips = EXCLUDED.mae_pips,
            early_mae_pips = EXCLUDED.early_mae_pips,
            time_to_mfe_minutes = EXCLUDED.time_to_mfe_minutes,
            time_to_mae_minutes = EXCLUDED.time_to_mae_minutes,
            pnl_60m_pips = EXCLUDED.pnl_60m_pips, pnl_240m_pips = EXCLUDED.pnl_240m_pips,
            pnl_1440m_pips = EXCLUDED.pnl_1440m_pips,
            ref_outcome = EXCLUDED.ref_outcome, ref_pnl_pips = EXCLUDED.ref_pnl_pips,
            time_to_tp_minutes = EXCLUDED.time_to_tp_minutes,
            time_to_sl_minutes = EXCLUDED.time_to_sl_minutes,
            candles_evaluated = EXCLUDED.candles_evaluated,
            status = EXCLUDED.status, evaluated_until = EXCLUDED.evaluated_until,
            updated_at = now()
    """)

    def save_rejection(self, rej: dict, outcome: dict):
        row = dict(outcome)
        row["rejection_id"] = row.pop("alert_id")
        # gate metadata not produced by simulate()
        row["stage"] = rej["stage"]
        row["reason"] = rej.get("reason")
        row["session"] = rej.get("session")
        row["hour_utc"] = rej.get("hour_utc")
        # simulate() emits environment; rejection_outcomes has no such column
        row.pop("environment", None)
        with self.forex.begin() as conn:
            conn.execute(self.UPSERT, row)

    # -- run ----------------------------------------------------------------

    def run(self, days_back: int = 7, strategy: str = DEFAULT_STRATEGY, dry_run: bool = False) -> dict:
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)

        df = self.get_candidate_rejections(start, end, strategy)
        if df.empty:
            logger.info("No candidate %s rejections to analyze", strategy)
            return {"analyzed": 0, "resolved": 0, "open": 0, "no_data": 0}

        stats = {"analyzed": 0, "resolved": 0, "open": 0, "no_data": 0}
        for rej in df.to_dict("records"):
            try:
                outcome = self.simulate(self._rejection_to_sig(rej))
            except Exception as e:
                logger.error("simulate failed for rejection %s: %s", rej["rejection_id"], e)
                continue
            if not dry_run:
                self.save_rejection(rej, outcome)
            stats["analyzed"] += 1
            stats[{"RESOLVED": "resolved", "OPEN": "open", "NO_DATA": "no_data"}[outcome["status"]]] += 1

        logger.info(
            "Gate-edge run (%s): %d analyzed (%d resolved, %d open, %d no-data)%s",
            strategy, stats["analyzed"], stats["resolved"], stats["open"], stats["no_data"],
            " [dry-run]" if dry_run else "",
        )
        return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Per-gate rejection edge analyzer (non-SMC strategies)")
    ap.add_argument("--days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY, help="Strategy to analyze (strategy_rejections.strategy)")
    ap.add_argument("--dry-run", action="store_true", help="Simulate but do not save")
    args = ap.parse_args()
    GateEdgeAnalyzer().run(days_back=args.days, strategy=args.strategy, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
