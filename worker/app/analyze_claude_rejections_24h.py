"""Evaluate the last 24h of Claude rejections with detailed per-alert outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

FOREX_DSN = "host=postgres dbname=forex user=postgres password=postgres"
STRAT_DSN = "host=postgres dbname=strategy_config user=postgres password=postgres"

TIMEOUT_HOURS = 24
DEFAULT_SL_PIPS = 10.0
DEFAULT_TP_PIPS = 15.0


def pip_size(epic: str) -> float:
    if "CFEGOLD" in epic:
        return 0.1
    if "JPY" in epic:
        return 0.01
    return 0.0001


def load_pair_configs() -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with psycopg2.connect(STRAT_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips "
            "FROM smc_simple_pair_overrides"
        )
        for epic, sl, tp in cur.fetchall():
            if sl is None or tp is None:
                continue
            out.setdefault(epic, (float(sl), float(tp)))
    return out


@dataclass
class Result:
    alert_id: int
    ts: str
    epic: str
    signal_type: str
    entry: float
    sl: float
    tp: float
    sl_pips: float
    tp_pips: float
    confidence: float
    claude_score: Optional[int]
    outcome: str
    pips: float
    minutes_to_resolve: Optional[int]
    max_favorable_pips: float
    max_adverse_pips: float
    current_pnl_pips: float
    reason: str = ""


def simulate(cur, r) -> Result:
    epic = r["epic"]
    signal = r["signal_type"]
    entry = float(r["price"])
    ts = r["alert_timestamp"]
    pip = pip_size(epic)

    sl_pips = r["sl_pips"]
    tp_pips = r["tp_pips"]

    is_long = signal.upper() in ("BULL", "BUY", "LONG")
    if is_long:
        sl = entry - sl_pips * pip
        tp = entry + tp_pips * pip
    else:
        sl = entry + sl_pips * pip
        tp = entry - tp_pips * pip

    cur.execute(
        """
        SELECT start_time, high, low, close FROM ig_candles
        WHERE epic = %s AND timeframe = 1
          AND start_time >= %s AND start_time <= %s
        ORDER BY start_time ASC
        """,
        (epic, ts, ts + timedelta(hours=TIMEOUT_HOURS)),
    )
    candles = cur.fetchall()

    max_fav = 0.0
    max_adv = 0.0
    last_close = entry
    outcome = "OPEN"
    pips = 0.0
    minutes: Optional[int] = None

    for c in candles:
        hi = float(c["high"])
        lo = float(c["low"])
        last_close = float(c["close"])

        if is_long:
            max_fav = max(max_fav, (hi - entry) / pip)
            max_adv = min(max_adv, (lo - entry) / pip)
        else:
            max_fav = max(max_fav, (entry - lo) / pip)
            max_adv = min(max_adv, (entry - hi) / pip)

        if outcome == "OPEN":
            if is_long:
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

            if hit_sl and hit_tp:
                outcome, pips = "LOSS", -sl_pips
            elif hit_sl:
                outcome, pips = "LOSS", -sl_pips
            elif hit_tp:
                outcome, pips = "WIN", tp_pips
            if outcome != "OPEN":
                minutes = int((c["start_time"] - ts).total_seconds() / 60)

    if outcome == "OPEN":
        if not candles:
            outcome = "NO_DATA"
        else:
            outcome = "STILL_OPEN"

    if is_long:
        current_pnl = (last_close - entry) / pip
    else:
        current_pnl = (entry - last_close) / pip

    return Result(
        r["id"], r["alert_timestamp"].strftime("%Y-%m-%d %H:%M"),
        epic, signal, entry, sl, tp, sl_pips, tp_pips,
        float(r["confidence_score"]), r["claude_score"],
        outcome, pips, minutes,
        round(max_fav, 1), round(max_adv, 1), round(current_pnl, 1),
        r.get("reason", ""),
    )


def main() -> None:
    pair_cfg = load_pair_configs()

    with psycopg2.connect(FOREX_DSN) as conn:
        conn.set_session(readonly=True)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, alert_timestamp, epic, signal_type, price,
                   confidence_score, claude_score,
                   LEFT(COALESCE(claude_reason, ''), 200) AS reason
            FROM alert_history
            WHERE claude_decision = 'REJECT'
              AND alert_timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY alert_timestamp ASC
            """
        )
        rows = cur.fetchall()
        results: list[Result] = []
        for row in rows:
            sl_pips, tp_pips = pair_cfg.get(row["epic"], (DEFAULT_SL_PIPS, DEFAULT_TP_PIPS))
            row["sl_pips"] = sl_pips
            row["tp_pips"] = tp_pips
            results.append(simulate(cur, row))

    print(f"\n{'=' * 100}")
    print(f"LAST 24H CLAUDE REJECTIONS — candle-validated outcomes (per-pair SL/TP, {TIMEOUT_HOURS}h window)")
    print(f"{'=' * 100}\n")

    header = (
        f"{'ID':>5} {'Time (UTC)':<17} {'Pair':<22} {'Dir':<5} {'Score':>5} "
        f"{'Conf':>5} {'Entry':>10} {'SL/TP':<8} {'MFE':>6} {'MAE':>6} "
        f"{'Outcome':<11} {'Pips':>6} {'Mins':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        sl_tp = f"{int(r.sl_pips)}/{int(r.tp_pips)}"
        mins = "-" if r.minutes_to_resolve is None else str(r.minutes_to_resolve)
        print(
            f"{r.alert_id:>5} {r.ts:<17} {r.epic:<22} {r.signal_type:<5} "
            f"{(r.claude_score or 0):>5} {r.confidence:>5.2f} {r.entry:>10.5f} "
            f"{sl_tp:<8} {r.max_favorable_pips:>+6.1f} {r.max_adverse_pips:>+6.1f} "
            f"{r.outcome:<11} {r.pips:>+6.1f} {mins:>5}"
        )

    resolved = [r for r in results if r.outcome in ("WIN", "LOSS")]
    wins = [r for r in resolved if r.outcome == "WIN"]
    losses = [r for r in resolved if r.outcome == "LOSS"]
    opens = [r for r in results if r.outcome == "STILL_OPEN"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rejections:          {len(results)}")
    print(f"  Resolved (hit SL or TP): {len(resolved)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Still open:              {len(opens)}")

    if resolved:
        net = sum(r.pips for r in resolved)
        gw = sum(r.pips for r in wins)
        gl = abs(sum(r.pips for r in losses))
        pf = gw / gl if gl else float("inf")
        wr = 100.0 * len(wins) / len(resolved)
        print(f"\nResolved metrics:")
        print(f"  Win rate:   {wr:.1f}%")
        print(f"  Net pips:   {net:+.1f}")
        print(f"  PF:         {pf:.2f}")
        verdict = "CLAUDE WAS RIGHT" if net < 0 else ("CLAUDE WAS WRONG" if net > 0 else "NEUTRAL")
        print(f"  Verdict:    {verdict}")

    if opens:
        open_mfe = [r.max_favorable_pips for r in opens]
        open_mae = [r.max_adverse_pips for r in opens]
        print(f"\nStill-open rejections (excursion so far):")
        for r in opens:
            print(
                f"  #{r.alert_id} {r.epic} {r.signal_type}: "
                f"MFE +{r.max_favorable_pips} / MAE {r.max_adverse_pips} / current {r.current_pnl_pips:+.1f} pips"
            )

    print(f"\n{'=' * 60}")
    print("REASON VALIDITY CHECK (did the rejection rationale play out?)")
    print("=" * 60)
    for r in results:
        verdict = {
            "WIN": "❌ CLAUDE WRONG",
            "LOSS": "✅ CLAUDE RIGHT",
            "STILL_OPEN": "⏳ Open",
            "NO_DATA": "- no data",
        }.get(r.outcome, "?")
        print(f"\n#{r.alert_id} {r.epic} {r.signal_type} @{r.ts}  →  {verdict} ({r.pips:+.1f}p, MFE +{r.max_favorable_pips}/MAE {r.max_adverse_pips})")
        print(f"  Reason: {r.reason[:180]}")


if __name__ == "__main__":
    main()
