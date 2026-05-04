"""Shadow analysis for direction-aware cooldown bypass.

For every alert in alert_history that did NOT result in a trade, ask:
  - Was it blocked by the existing per-epic cooldown? (proxy: prior closed
    trade on same epic within cooldown window before alert_timestamp)
  - Would the proposed direction-aware bypass have let it through?
    Bypass requires:
      * opposite direction to the prior trade
      * prior trade was a loser (profit_loss < 0)
      * no currently-open position on the epic at alert time
  - For bypass-eligible alerts, walk forward through 1m candles to see whether
    a hypothetical entry at alert price with the strategy's SL/TP would have
    hit TP, SL, or timed out.

Run inside task-worker:
  docker exec -it task-worker python /app/forex_scanner/scripts/cooldown_reversal_shadow.py --days 90

Outputs:
  /tmp/cooldown_shadow_<utcdate>.csv  (per-alert detail)
  stdout                              (aggregate summary)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

# --- cooldown config (mirrors dev-app/config.py) -----------------------------
DEFAULT_COOLDOWN_MIN = 30
EPIC_SPECIFIC_COOLDOWNS = {
    "CS.D.EURUSD.CEEM.IP": 45,
    "CS.D.GBPUSD.MINI.IP": 45,
    "CS.D.USDJPY.MINI.IP": 30,
    "CS.D.AUDUSD.MINI.IP": 30,
    "CS.D.USDCAD.MINI.IP": 30,
}

# --- SL/TP per strategy (pips) used for hypothetical fills -------------------
# These are intentionally conservative defaults; the order executor uses
# database-driven values that can vary by pair/regime/scalp flag. For the
# shadow run we only need a stable approximation.
STRATEGY_SLTP_PIPS = {
    "SMC_SIMPLE": (12, 30),
    "RANGE_FADE": (8, 12),
    "MEAN_REVERSION": (12, 18),
    "XAU_GOLD": (80, 160),
}
DEFAULT_SLTP_PIPS = (10, 20)

JPY_EPICS = ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "CADJPY", "NZDJPY")


def pip_size(epic: str) -> float:
    if "CFEGOLD" in epic:
        return 0.1
    if any(p in epic for p in JPY_EPICS):
        return 0.01
    return 0.0001


def normalize_dir(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    if s in ("BUY", "BULL", "LONG"):
        return "BUY"
    if s in ("SELL", "BEAR", "SHORT"):
        return "SELL"
    return None


@dataclass
class PriorTrade:
    id: int
    direction: str
    profit_loss: Optional[float]
    closed_at: datetime
    status: str


def fetch_blocked_alerts(cur, days: int):
    """Pull alerts that have no executed trade and have a recent prior trade."""
    cur.execute(
        """
        SELECT a.id, a.alert_timestamp, a.epic, a.signal_type, a.strategy,
               a.price, a.confidence_score
        FROM alert_history a
        LEFT JOIN trade_log t ON t.alert_id = a.id
        WHERE a.alert_timestamp >= NOW() - INTERVAL '%s days'
          AND t.id IS NULL
          AND a.claude_approved = TRUE
        ORDER BY a.alert_timestamp ASC
        """,
        (days,),
    )
    return cur.fetchall()


def fetch_prior_trade(cur, epic: str, alert_ts: datetime, cooldown_min: int) -> Optional[PriorTrade]:
    cur.execute(
        """
        SELECT id, direction, profit_loss, closed_at, status
        FROM trade_log
        WHERE symbol = %s
          AND status IN ('closed', 'expired')
          AND closed_at IS NOT NULL
          AND closed_at <= %s
          AND closed_at >= %s
        ORDER BY closed_at DESC
        LIMIT 1
        """,
        (epic, alert_ts, alert_ts - timedelta(minutes=cooldown_min)),
    )
    row = cur.fetchone()
    if not row:
        return None
    return PriorTrade(
        id=row["id"],
        direction=normalize_dir(row["direction"]) or "",
        profit_loss=float(row["profit_loss"]) if row["profit_loss"] is not None else None,
        closed_at=row["closed_at"],
        status=row["status"],
    )


def has_open_position(cur, epic: str, alert_ts: datetime) -> bool:
    cur.execute(
        """
        SELECT 1 FROM trade_log
        WHERE symbol = %s
          AND status = 'open'
          AND timestamp <= %s
          AND (closed_at IS NULL OR closed_at > %s)
        LIMIT 1
        """,
        (epic, alert_ts, alert_ts),
    )
    return cur.fetchone() is not None


def simulate_outcome(cur, epic: str, alert_ts: datetime, direction: str, entry: float,
                     sl_pips: float, tp_pips: float, timeout_min: int = 240):
    """Walk forward through 1m candles. Return (outcome, exit_price, minutes)."""
    p = pip_size(epic)
    if direction == "BUY":
        sl_price, tp_price = entry - sl_pips * p, entry + tp_pips * p
    else:
        sl_price, tp_price = entry + sl_pips * p, entry - tp_pips * p

    cur.execute(
        """
        SELECT start_time, high, low FROM ig_candles
        WHERE epic = %s AND timeframe = 1
          AND start_time > %s AND start_time <= %s
        ORDER BY start_time ASC
        """,
        (epic, alert_ts, alert_ts + timedelta(minutes=timeout_min)),
    )
    rows = cur.fetchall()
    for r in rows:
        hi, lo = float(r["high"]), float(r["low"])
        sl_hit = lo <= sl_price if direction == "BUY" else hi >= sl_price
        tp_hit = hi >= tp_price if direction == "BUY" else lo <= tp_price
        if sl_hit and tp_hit:
            # ambiguous within the same minute — assume SL hit first (worst case)
            mins = int((r["start_time"] - alert_ts).total_seconds() / 60)
            return ("sl", sl_price, mins, -sl_pips)
        if sl_hit:
            mins = int((r["start_time"] - alert_ts).total_seconds() / 60)
            return ("sl", sl_price, mins, -sl_pips)
        if tp_hit:
            mins = int((r["start_time"] - alert_ts).total_seconds() / 60)
            return ("tp", tp_price, mins, tp_pips)
    return ("timeout", None, timeout_min, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--db-host", default=os.getenv("DB_HOST", "postgres"))
    ap.add_argument("--db-name", default=os.getenv("DB_NAME", "forex"))
    ap.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"))
    ap.add_argument("--db-pass", default=os.getenv("DB_PASSWORD", "postgres"))
    ap.add_argument("--min-loss-pips", type=float, default=0.0,
                    help="Only count prior trades as 'losers' if loss exceeds this magnitude (proxy via profit_loss).")
    ap.add_argument("--require-different-strategy", action="store_true",
                    help="Only bypass when the new alert's strategy differs from the prior trade's alert strategy.")
    ap.add_argument("--out", default=f"/tmp/cooldown_shadow_{datetime.utcnow():%Y%m%d_%H%M}.csv")
    args = ap.parse_args()

    conn = psycopg2.connect(host=args.db_host, dbname=args.db_name,
                            user=args.db_user, password=args.db_pass)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    alerts = fetch_blocked_alerts(cur, args.days)
    print(f"[shadow] candidate alerts (no trade, claude-approved, last {args.days}d): {len(alerts)}")

    rows_out = []
    counts = defaultdict(int)
    pip_total = 0.0
    pip_wins = pip_losses = 0
    bypass_per_strategy = defaultdict(lambda: {"n": 0, "tp": 0, "sl": 0, "timeout": 0, "pips": 0.0})

    for a in alerts:
        epic = a["epic"]
        alert_ts = a["alert_timestamp"]
        new_dir = normalize_dir(a["signal_type"])
        if not new_dir:
            continue
        cd = EPIC_SPECIFIC_COOLDOWNS.get(epic, DEFAULT_COOLDOWN_MIN)
        prior = fetch_prior_trade(cur, epic, alert_ts, cd)
        if not prior:
            counts["no_prior_in_window"] += 1
            continue
        counts["blocked_by_cooldown"] += 1

        same_dir = prior.direction == new_dir
        was_loser = (prior.profit_loss is not None and prior.profit_loss < -args.min_loss_pips)
        if same_dir:
            counts["blocked_same_direction"] += 1
            continue
        if not was_loser:
            counts["blocked_prior_winner"] += 1
            continue
        if has_open_position(cur, epic, alert_ts):
            counts["blocked_open_position"] += 1
            continue

        counts["bypass_eligible"] += 1
        strategy = (a["strategy"] or "").upper()
        sl_p, tp_p = STRATEGY_SLTP_PIPS.get(strategy, DEFAULT_SLTP_PIPS)
        outcome, exit_px, mins, pips = simulate_outcome(
            cur, epic, alert_ts, new_dir, float(a["price"]), sl_p, tp_p
        )
        pip_total += pips
        if pips > 0:
            pip_wins += 1
        elif pips < 0:
            pip_losses += 1
        s = bypass_per_strategy[strategy]
        s["n"] += 1
        s[outcome] += 1
        s["pips"] += pips

        rows_out.append({
            "alert_id": a["id"],
            "alert_ts": alert_ts.isoformat(),
            "epic": epic,
            "strategy": strategy,
            "new_direction": new_dir,
            "alert_price": float(a["price"]),
            "confidence": float(a["confidence_score"]) if a["confidence_score"] else None,
            "prior_trade_id": prior.id,
            "prior_direction": prior.direction,
            "prior_pnl": prior.profit_loss,
            "prior_closed_at": prior.closed_at.isoformat(),
            "cooldown_min": cd,
            "sl_pips": sl_p, "tp_pips": tp_p,
            "outcome": outcome,
            "exit_price": exit_px,
            "elapsed_min": mins,
            "pips_pnl": pips,
        })

    # write csv
    if rows_out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    # summary
    print("\n=== shadow analysis summary ===")
    for k, v in counts.items():
        print(f"  {k:>30} : {v}")
    print()
    print(f"  bypass eligible           : {len(rows_out)}")
    if rows_out:
        wr = pip_wins / max(pip_wins + pip_losses, 1) * 100
        avg = pip_total / len(rows_out)
        print(f"  hypothetical wins         : {pip_wins}")
        print(f"  hypothetical losses       : {pip_losses}")
        print(f"  timeouts                  : {len(rows_out) - pip_wins - pip_losses}")
        print(f"  win rate                  : {wr:.1f}%")
        print(f"  total pips                : {pip_total:+.1f}")
        print(f"  avg pips / bypassed alert : {avg:+.2f}")
        print()
        print("  per-strategy breakdown:")
        for strat, s in sorted(bypass_per_strategy.items()):
            print(f"    {strat:<18} n={s['n']:>3}  tp={s['tp']:>3}  sl={s['sl']:>3}  to={s['timeout']:>3}  pips={s['pips']:+.1f}")
        print()
        print(f"  detail csv : {args.out}")
    else:
        print("  (no bypass-eligible alerts found)")


if __name__ == "__main__":
    sys.exit(main() or 0)
