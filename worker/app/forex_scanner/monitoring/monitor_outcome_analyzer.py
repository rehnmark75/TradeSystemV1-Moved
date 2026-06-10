"""
Monitor-Only Outcome Analyzer

Counterfactual forward-outcome layer for MONITOR-ONLY signals.

A monitor-only signal is one that passed strategy logic (so it is logged to
`alert_history`) but was never executed, so it has no trade_log row and no
outcome anywhere. We define this population operationally as "logged but not
executed" (no matching trade_log row) rather than via per-pair monitor_only
config flags, because those flags are inconsistent across strategies and do not
reliably reflect execution (e.g. strategies emit signals while is_enabled=false;
gold demo runs un-executed with monitor_only=false). Cooldown / risk / LPF
blocks never reach alert_history (they land in the rejection/validator tables),
so non-executed alert_history rows are exactly the monitor-only set.

This job walks `ig_candles` (1m) forward from each such signal and records what
price actually did:

    * MFE / MAE (max favorable / adverse excursion) — the spine. No stop/target
      needed, valid for every strategy. Answers: did it move our way at all, did
      losers go against us immediately (early_mae_pips), is TP too wide / SL too
      tight, which session / epic carries the edge.
    * Net move at 60m / 240m / 1440m horizons.
    * One FIXED reference SL/TP grid → exact HIT_TP/HIT_SL win-rate (a comparable
      anchor, NOT each strategy's native stop — those are not recoverable from
      alert_history and are deliberately not reconstructed).

Results upsert into `monitor_only_outcomes` (forex DB). OPEN rows (signal younger
than the horizon) are re-evaluated on subsequent runs until the window completes.

Usage:
    docker exec -it task-worker python /app/forex_scanner/monitoring/monitor_outcome_analyzer.py
    docker exec -it task-worker python /app/forex_scanner/monitoring/monitor_outcome_analyzer.py --days 30
    docker exec -it task-worker python /app/forex_scanner/monitoring/monitor_outcome_analyzer.py --dry-run

Invoked automatically once/24h by the live scanner via
TradingOrchestrator._run_monitor_outcome_analysis_if_due().

Created: 2026-06-09
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/forex")

# Archived / disabled strategies whose stray signals we don't bother simulating.
DEAD_STRATEGIES = {"RANGING_MARKET", "RANGE_STRUCTURE"}

HORIZON_MINUTES = 1440          # 24h forward evaluation window
SNAPSHOT_HORIZONS = [60, 240, 1440]
EARLY_WINDOW_MINUTES = 15       # window for early_mae_pips ("losers die on arrival")
# Max minutes the first post-signal candle may lag the signal before we refuse
# to reconstruct an entry from it (avoids misanchoring across a weekend/session
# gap, where the next candle could be hours/days later).
MAX_ENTRY_RECONSTRUCT_GAP_MINUTES = 10
# Don't evaluate a signal until it's at least this old, so a forward candle
# exists. Without this, a run firing seconds after a signal (e.g. right after a
# container restart, which re-arms the 24h hook) evaluates it to a premature
# NO_DATA. Must be >= the 1m candle cadence.
MIN_SIGNAL_AGE_MINUTES = 3

# Reference SL/TP grid (pips). A comparable diagnostic anchor for exact win-rate,
# scaled by instrument so it is meaningful for gold vs FX. NOT a native stop.
REF_GRID_DEFAULT = (10.0, 15.0)
REF_GRID_BY_KIND = {
    "JPY": (10.0, 15.0),
    "GOLD": (80.0, 160.0),
}

def pip_multiplier(epic: str) -> int:
    """Pips per unit price move. 100 for JPY, 10 for gold, 10000 otherwise."""
    e = (epic or "").upper()
    if "JPY" in e:
        return 100
    if "CFEGOLD" in e or "XAU" in e or "GOLD" in e:
        return 10
    return 10000


def instrument_kind(epic: str) -> str:
    e = (epic or "").upper()
    if "JPY" in e:
        return "JPY"
    if "CFEGOLD" in e or "XAU" in e or "GOLD" in e:
        return "GOLD"
    return "FX"


def ref_grid(epic: str):
    return REF_GRID_BY_KIND.get(instrument_kind(epic), REF_GRID_DEFAULT)


def normalize_direction(signal_type: str):
    """Map any signal_type encoding to 'BUY' / 'SELL'."""
    s = (signal_type or "").upper()
    if s in ("BULL", "BUY", "LONG"):
        return "BUY"
    if s in ("BEAR", "SELL", "SHORT"):
        return "SELL"
    return None


class MonitorOutcomeAnalyzer:
    def __init__(self):
        self.forex = create_engine(DATABASE_URL)

    # -- signal selection ---------------------------------------------------

    def get_candidate_signals(self, start_date, end_date) -> pd.DataFrame:
        """Monitor-only (logged-but-not-executed) alert_history rows to evaluate.

        A signal qualifies if it has no trade_log execution and is old enough to
        have a forward candle. Rows with no outcome yet, an OPEN outcome, or a
        NO_DATA outcome are included (re-evaluated until their horizon completes
        or candles finally arrive). Archived strategies are skipped.
        """
        max_signal_time = end_date - timedelta(minutes=MIN_SIGNAL_AGE_MINUTES)
        q = """
            SELECT a.id AS alert_id, a.strategy, a.epic, a.pair, a.environment,
                   a.signal_type, a.alert_timestamp, a.price, a.bid_price, a.ask_price
            FROM alert_history a
            LEFT JOIN trade_log t          ON t.alert_id = a.id
            LEFT JOIN monitor_only_outcomes o ON o.alert_id = a.id
            WHERE a.alert_timestamp >= :start_date
              AND a.alert_timestamp <  :end_date
              -- Skip signals too fresh to have a forward 1m candle yet, else they
              -- evaluate to a premature NO_DATA.
              AND a.alert_timestamp <= :max_signal_time
              AND a.strategy IS NOT NULL
              AND NOT (a.strategy = ANY(:dead))
              -- NOTE: zero/NULL price is NOT excluded. raw_monitor_signal rows
              -- persist with price=0 (and bid/ask = ±half-spread around 0, so the
              -- mid is also 0); their true entry is reconstructed from the first
              -- post-signal candle open in simulate(). Excluding them here dropped
              -- ~160 monitor-only signals (notably RANGE_FADE / FA_OR_ATR_TRAIL /
              -- SMC_MOMENTUM) from outcome analysis entirely.
              AND t.alert_id IS NULL
              -- Re-select OPEN (horizon incomplete) and NO_DATA (candles may have
              -- arrived since the first attempt); only finished RESOLVED is skipped.
              AND (o.id IS NULL OR o.status IN ('OPEN', 'NO_DATA'))
            ORDER BY a.alert_timestamp ASC
        """
        with self.forex.connect() as conn:
            df = pd.DataFrame(
                conn.execute(
                    text(q),
                    {
                        "start_date": start_date, "end_date": end_date,
                        "max_signal_time": max_signal_time, "dead": list(DEAD_STRATEGIES),
                    },
                ).mappings().all()
            )
        return df

    def fetch_candles(self, epic, start_time, end_time) -> pd.DataFrame:
        q = """
            SELECT start_time AS timestamp, open, high, low, close
            FROM ig_candles
            WHERE epic = :epic AND timeframe = 1
              AND start_time >  :start_time
              AND start_time <= :end_time
            ORDER BY start_time ASC
        """
        with self.forex.connect() as conn:
            return pd.DataFrame(
                conn.execute(text(q), {"epic": epic, "start_time": start_time, "end_time": end_time}).mappings().all()
            )

    # -- simulation ---------------------------------------------------------

    def simulate(self, sig: dict) -> dict:
        epic = sig["epic"]
        direction = normalize_direction(sig["signal_type"])
        signal_ts = pd.to_datetime(sig["alert_timestamp"]).to_pydatetime()
        mult = pip_multiplier(epic)
        ref_sl_pips, ref_tp_pips = ref_grid(epic)

        # Entry: BUY fills at ask, SELL at bid; fall back to mid (price).
        # raw_monitor_signal rows carry price=0 with bid/ask = ±half-spread
        # around 0 (mid==0), so none of these columns give a usable entry. In
        # that case leave entry=None and reconstruct it from the first
        # post-signal candle open once candles are fetched (below).
        price = float(sig["price"]) if sig.get("price") else 0.0
        ask = float(sig["ask_price"]) if sig.get("ask_price") else None
        bid = float(sig["bid_price"]) if sig.get("bid_price") else None
        if price > 0:
            entry = (ask if (direction == "BUY" and ask) else
                     bid if (direction == "SELL" and bid) else price)
        else:
            entry = None  # reconstructed from candles after fetch

        out = {
            "alert_id": sig["alert_id"], "strategy": sig["strategy"], "epic": epic,
            "pair": sig.get("pair"), "environment": sig.get("environment"),
            "signal_timestamp": signal_ts, "direction": direction, "entry_price": entry,
            "pip_multiplier": mult, "horizon_minutes": HORIZON_MINUTES,
            "ref_sl_pips": ref_sl_pips, "ref_tp_pips": ref_tp_pips,
            "mfe_pips": None, "mae_pips": None, "early_mae_pips": None,
            "time_to_mfe_minutes": None, "time_to_mae_minutes": None,
            "pnl_60m_pips": None, "pnl_240m_pips": None, "pnl_1440m_pips": None,
            "ref_outcome": None, "ref_pnl_pips": None,
            "time_to_tp_minutes": None, "time_to_sl_minutes": None,
            "candles_evaluated": 0, "status": "NO_DATA",
            "evaluated_until": None,
        }

        if direction is None:
            out["status"] = "NO_DATA"
            return out

        now = datetime.utcnow()
        horizon_end = signal_ts + timedelta(minutes=HORIZON_MINUTES)
        fetch_end = min(horizon_end, now)
        candles = self.fetch_candles(epic, signal_ts, fetch_end)
        if candles.empty:
            return out

        # Reconstruct entry for rows with no usable stored price (price=0
        # monitor-only signals): realistic next-bar fill at the first
        # post-signal candle open.
        if entry is None:
            first_ts = pd.to_datetime(candles.iloc[0]["timestamp"]).to_pydatetime()
            gap_min = (first_ts - signal_ts).total_seconds() / 60.0
            if gap_min > MAX_ENTRY_RECONSTRUCT_GAP_MINUTES:
                # First candle lags the signal too far (weekend/session gap) to
                # trust as the entry fill; don't fabricate a misanchored entry.
                out["status"] = "NO_DATA"
                return out
            entry = float(candles.iloc[0]["open"])
            out["entry_price"] = entry

        pip = 1.0 / mult
        sl_price = entry - ref_sl_pips * pip if direction == "BUY" else entry + ref_sl_pips * pip
        tp_price = entry + ref_tp_pips * pip if direction == "BUY" else entry - ref_tp_pips * pip

        mfe = mae = 0.0
        t_mfe = t_mae = None
        early_mae = 0.0
        snap = {h: None for h in SNAPSHOT_HORIZONS}
        ref_hit = None  # ('HIT_TP'|'HIT_SL', minutes)

        for c in candles.itertuples(index=False):
            ts = pd.to_datetime(c.timestamp).to_pydatetime()
            elapsed = (ts - signal_ts).total_seconds() / 60.0
            high, low, close = float(c.high), float(c.low), float(c.close)

            if direction == "BUY":
                fav = (high - entry) * mult
                adv = (entry - low) * mult
                net = (close - entry) * mult
            else:
                fav = (entry - low) * mult
                adv = (high - entry) * mult
                net = (entry - close) * mult

            if fav > mfe:
                mfe, t_mfe = fav, int(round(elapsed))
            if adv > mae:
                mae, t_mae = adv, int(round(elapsed))
            if elapsed <= EARLY_WINDOW_MINUTES and adv > early_mae:
                early_mae = adv

            for h in SNAPSHOT_HORIZONS:
                if elapsed <= h:
                    snap[h] = net  # last close within the boundary wins

            # Reference grid: first touch, pessimistic SL-first within the candle.
            if ref_hit is None:
                if direction == "BUY":
                    if low <= sl_price:
                        ref_hit = ("HIT_SL", int(round(elapsed)))
                    elif high >= tp_price:
                        ref_hit = ("HIT_TP", int(round(elapsed)))
                else:
                    if high >= sl_price:
                        ref_hit = ("HIT_SL", int(round(elapsed)))
                    elif low <= tp_price:
                        ref_hit = ("HIT_TP", int(round(elapsed)))

        last_ts = pd.to_datetime(candles.iloc[-1]["timestamp"]).to_pydatetime()
        last_elapsed = (last_ts - signal_ts).total_seconds() / 60.0
        # A horizon snapshot is only valid once the window actually reaches it;
        # otherwise snap[h] holds the latest net and would be mislabeled.
        for h in SNAPSHOT_HORIZONS:
            if last_elapsed < h:
                snap[h] = None

        out["candles_evaluated"] = int(len(candles))
        out["mfe_pips"] = round(mfe, 2)
        out["mae_pips"] = round(mae, 2)
        out["early_mae_pips"] = round(early_mae, 2)
        out["time_to_mfe_minutes"] = t_mfe
        out["time_to_mae_minutes"] = t_mae
        out["pnl_60m_pips"] = None if snap[60] is None else round(snap[60], 2)
        out["pnl_240m_pips"] = None if snap[240] is None else round(snap[240], 2)
        out["pnl_1440m_pips"] = None if snap[1440] is None else round(snap[1440], 2)
        out["evaluated_until"] = last_ts

        window_complete = now >= horizon_end
        if ref_hit is not None:
            out["ref_outcome"] = ref_hit[0]
            if ref_hit[0] == "HIT_TP":
                out["ref_pnl_pips"] = ref_tp_pips
                out["time_to_tp_minutes"] = ref_hit[1]
            else:
                out["ref_pnl_pips"] = -ref_sl_pips
                out["time_to_sl_minutes"] = ref_hit[1]
            # Outcome is fixed once touched, but MFE/MAE keep growing until the
            # horizon, so only finalize when the full window has elapsed.
            out["status"] = "RESOLVED" if window_complete else "OPEN"
        elif window_complete:
            out["ref_outcome"] = "TIMEOUT"
            out["ref_pnl_pips"] = round(snap[1440], 2) if snap[1440] is not None else None
            out["status"] = "RESOLVED"
        else:
            out["status"] = "OPEN"

        return out

    # -- persistence --------------------------------------------------------

    UPSERT = text("""
        INSERT INTO monitor_only_outcomes (
            alert_id, strategy, epic, pair, environment, signal_timestamp,
            direction, entry_price, pip_multiplier, horizon_minutes,
            mfe_pips, mae_pips, early_mae_pips, time_to_mfe_minutes, time_to_mae_minutes,
            pnl_60m_pips, pnl_240m_pips, pnl_1440m_pips,
            ref_sl_pips, ref_tp_pips, ref_outcome, ref_pnl_pips,
            time_to_tp_minutes, time_to_sl_minutes,
            candles_evaluated, status, evaluated_until, updated_at
        ) VALUES (
            :alert_id, :strategy, :epic, :pair, :environment, :signal_timestamp,
            :direction, :entry_price, :pip_multiplier, :horizon_minutes,
            :mfe_pips, :mae_pips, :early_mae_pips, :time_to_mfe_minutes, :time_to_mae_minutes,
            :pnl_60m_pips, :pnl_240m_pips, :pnl_1440m_pips,
            :ref_sl_pips, :ref_tp_pips, :ref_outcome, :ref_pnl_pips,
            :time_to_tp_minutes, :time_to_sl_minutes,
            :candles_evaluated, :status, :evaluated_until, now()
        )
        ON CONFLICT (alert_id) DO UPDATE SET
            -- entry_price can go NULL -> reconstructed when a row that first
            -- evaluated to NO_DATA (no candles yet) is later re-evaluated.
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

    def save(self, outcome: dict):
        with self.forex.begin() as conn:
            conn.execute(self.UPSERT, outcome)

    # -- run ----------------------------------------------------------------

    def run(self, days_back: int = 7, dry_run: bool = False) -> dict:
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)

        df = self.get_candidate_signals(start, end)
        if df.empty:
            logger.info("No candidate monitor-only signals to analyze")
            return {"analyzed": 0, "resolved": 0, "open": 0, "no_data": 0}

        stats = {"analyzed": 0, "resolved": 0, "open": 0, "no_data": 0}
        for sig in df.to_dict("records"):
            try:
                outcome = self.simulate(sig)
            except Exception as e:
                logger.error(f"simulate failed for alert {sig['alert_id']}: {e}")
                continue
            if not dry_run:
                self.save(outcome)
            stats["analyzed"] += 1
            stats[{"RESOLVED": "resolved", "OPEN": "open", "NO_DATA": "no_data"}[outcome["status"]]] += 1

        logger.info(
            f"Monitor-outcome run: {stats['analyzed']} analyzed "
            f"({stats['resolved']} resolved, {stats['open']} open, {stats['no_data']} no-data)"
            + (" [dry-run]" if dry_run else "")
        )
        return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Monitor-only outcome analyzer")
    ap.add_argument("--days", type=int, default=7, help="Lookback window in days")
    ap.add_argument("--dry-run", action="store_true", help="Simulate but do not save")
    args = ap.parse_args()
    MonitorOutcomeAnalyzer().run(days_back=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
