"""Walk forward through candle data to evaluate if Claude rejections would have been wins or losses.

Simulates the actual SMC Simple entry using per-pair SL/TP and determines which came first
in the post-rejection candle stream. Outputs aggregate stats by pair, claude score, confidence.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras

FOREX_DSN = "host=postgres dbname=forex user=postgres password=postgres"
STRAT_DSN = "host=postgres dbname=strategy_config user=postgres password=postgres"

TIMEOUT_HOURS = 48
DEFAULT_SL_PIPS = 10.0
DEFAULT_TP_PIPS = 15.0


def pip_size(epic: str) -> float:
    if "JPY" in epic or "GOLD" in epic or "CFEGOLD" in epic:
        if "CFEGOLD" in epic:
            return 0.1
        return 0.01
    return 0.0001


def load_pair_configs() -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with psycopg2.connect(STRAT_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips "
            "FROM smc_simple_pair_overrides WHERE is_enabled = TRUE OR is_enabled IS NULL"
        )
        for epic, sl, tp in cur.fetchall():
            if sl is None or tp is None:
                continue
            out.setdefault(epic, (float(sl), float(tp)))
    return out


@dataclass
class Result:
    alert_id: int
    epic: str
    signal_type: str
    entry: float
    sl: float
    tp: float
    sl_pips: float
    tp_pips: float
    confidence: float
    claude_score: Optional[int]
    outcome: str  # "WIN" | "LOSS" | "TIMEOUT" | "NO_DATA"
    pips: float
    minutes_to_resolve: Optional[int]
    reason_snippet: str = ""


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
        SELECT start_time, high, low, timeframe FROM ig_candles
        WHERE epic = %s AND timeframe = 1
          AND start_time >= %s AND start_time <= %s
        ORDER BY start_time ASC
        """,
        (epic, ts, ts + timedelta(hours=TIMEOUT_HOURS)),
    )
    candles = cur.fetchall()

    if not candles:
        cur.execute(
            """
            SELECT start_time, high, low, timeframe FROM ig_candles
            WHERE epic = %s AND timeframe = 5
              AND start_time >= %s AND start_time <= %s
            ORDER BY start_time ASC
            """,
            (epic, ts, ts + timedelta(hours=TIMEOUT_HOURS)),
        )
        candles = cur.fetchall()

    if not candles:
        return Result(
            r["id"], epic, signal, entry, sl, tp, sl_pips, tp_pips,
            float(r["confidence_score"]), r["claude_score"],
            "NO_DATA", 0.0, None, r.get("reason_snippet", ""),
        )

    for c in candles:
        hi = float(c["high"])
        lo = float(c["low"])
        # Conservative ordering: if both in same bar, assume SL hits first (worst case for Claude-reject "missed win" analysis).
        if is_long:
            if lo <= sl and hi >= tp:
                outcome, pips = "LOSS", -sl_pips
            elif lo <= sl:
                outcome, pips = "LOSS", -sl_pips
            elif hi >= tp:
                outcome, pips = "WIN", tp_pips
            else:
                continue
        else:
            if hi >= sl and lo <= tp:
                outcome, pips = "LOSS", -sl_pips
            elif hi >= sl:
                outcome, pips = "LOSS", -sl_pips
            elif lo <= tp:
                outcome, pips = "WIN", tp_pips
            else:
                continue

        minutes = int((c["start_time"] - ts).total_seconds() / 60)
        return Result(
            r["id"], epic, signal, entry, sl, tp, sl_pips, tp_pips,
            float(r["confidence_score"]), r["claude_score"],
            outcome, pips, minutes, r.get("reason_snippet", ""),
        )

    return Result(
        r["id"], epic, signal, entry, sl, tp, sl_pips, tp_pips,
        float(r["confidence_score"]), r["claude_score"],
        "TIMEOUT", 0.0, None, r.get("reason_snippet", ""),
    )


def summarize(results: list[Result]) -> None:
    resolved = [r for r in results if r.outcome in ("WIN", "LOSS")]
    wins = [r for r in resolved if r.outcome == "WIN"]
    losses = [r for r in resolved if r.outcome == "LOSS"]
    timeouts = [r for r in results if r.outcome == "TIMEOUT"]
    no_data = [r for r in results if r.outcome == "NO_DATA"]

    total = len(results)
    print(f"\n{'=' * 72}")
    print(f"CLAUDE REJECTION OUTCOME ANALYSIS")
    print(f"{'=' * 72}")
    print(f"Total rejected alerts analyzed: {total}")
    print(f"  Resolved (hit SL or TP within {TIMEOUT_HOURS}h): {len(resolved)}")
    print(f"  Timeout (neither hit):                          {len(timeouts)}")
    print(f"  No candle data:                                 {len(no_data)}")

    if resolved:
        total_pips = sum(r.pips for r in resolved)
        wr = 100.0 * len(wins) / len(resolved)
        gross_win = sum(r.pips for r in wins)
        gross_loss = abs(sum(r.pips for r in losses))
        pf = gross_win / gross_loss if gross_loss else float("inf")
        print(f"\nOverall (resolved only):")
        print(f"  Win rate:             {wr:.1f}%   ({len(wins)}W / {len(losses)}L)")
        print(f"  Net pips (missed):    {total_pips:+.1f}")
        print(f"  Profit factor:        {pf:.2f}")
        print(f"  Avg win:   {gross_win / max(1, len(wins)):.1f} pips")
        print(f"  Avg loss:  {-gross_loss / max(1, len(losses)):.1f} pips")

    # By pair
    print(f"\n{'-' * 72}")
    print("BY PAIR (Claude was right to REJECT if loss rate > 50% with R:R 1.5+)")
    print(f"{'-' * 72}")
    print(f"{'Pair':<24} {'N':>4} {'WR%':>6} {'Pips':>8} {'PF':>6} {'Verdict':<20}")
    buckets: dict[str, list[Result]] = defaultdict(list)
    for r in resolved:
        buckets[r.epic].append(r)
    for epic in sorted(buckets):
        rs = buckets[epic]
        w = sum(1 for x in rs if x.outcome == "WIN")
        l = len(rs) - w
        pips = sum(x.pips for x in rs)
        gw = sum(x.pips for x in rs if x.outcome == "WIN")
        gl = abs(sum(x.pips for x in rs if x.outcome == "LOSS"))
        pf = gw / gl if gl else float("inf")
        wr = 100.0 * w / len(rs)
        verdict = (
            "CLAUDE CORRECT" if pips < 0 else
            ("CLAUDE WRONG" if pips > 0 else "NEUTRAL")
        )
        print(f"{epic:<24} {len(rs):>4} {wr:>6.1f} {pips:>+8.1f} {pf:>6.2f} {verdict:<20}")

    # By signal type
    print(f"\n{'-' * 72}")
    print("BY SIGNAL DIRECTION")
    print(f"{'-' * 72}")
    dir_buckets: dict[str, list[Result]] = defaultdict(list)
    for r in resolved:
        d = "LONG" if r.signal_type.upper() in ("BULL", "BUY", "LONG") else "SHORT"
        dir_buckets[d].append(r)
    for d, rs in dir_buckets.items():
        w = sum(1 for x in rs if x.outcome == "WIN")
        pips = sum(x.pips for x in rs)
        print(f"  {d:<6} {len(rs):>4}   WR {100.0*w/len(rs):.1f}%   Net {pips:+.1f} pips")

    # By Claude score
    print(f"\n{'-' * 72}")
    print("BY CLAUDE SCORE (lower = more confidently rejected)")
    print(f"{'-' * 72}")
    score_buckets: dict[int, list[Result]] = defaultdict(list)
    for r in resolved:
        if r.claude_score is not None:
            score_buckets[r.claude_score].append(r)
    for s in sorted(score_buckets):
        rs = score_buckets[s]
        w = sum(1 for x in rs if x.outcome == "WIN")
        pips = sum(x.pips for x in rs)
        wr = 100.0 * w / len(rs)
        print(f"  Score {s}: n={len(rs):>3}   WR {wr:>5.1f}%   Net {pips:+7.1f} pips")

    # By confidence bucket
    print(f"\n{'-' * 72}")
    print("BY STRATEGY CONFIDENCE")
    print(f"{'-' * 72}")
    conf_buckets: dict[str, list[Result]] = defaultdict(list)
    for r in resolved:
        c = r.confidence
        if c < 0.55:
            k = "<0.55"
        elif c < 0.60:
            k = "0.55-0.60"
        elif c < 0.65:
            k = "0.60-0.65"
        elif c < 0.70:
            k = "0.65-0.70"
        else:
            k = ">=0.70"
        conf_buckets[k].append(r)
    for k in ["<0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", ">=0.70"]:
        rs = conf_buckets.get(k, [])
        if not rs:
            continue
        w = sum(1 for x in rs if x.outcome == "WIN")
        pips = sum(x.pips for x in rs)
        wr = 100.0 * w / len(rs)
        print(f"  {k:<12} n={len(rs):>3}   WR {wr:>5.1f}%   Net {pips:+7.1f} pips")

    # Top 10 biggest missed wins
    wins_sorted = sorted(wins, key=lambda r: -r.pips)[:10]
    if wins_sorted:
        print(f"\n{'-' * 72}")
        print("TOP 10 'CLAUDE WAS WRONG' (signals Claude rejected that hit TP)")
        print(f"{'-' * 72}")
        print(f"{'ID':>5} {'Pair':<22} {'Type':<5} {'Score':>5} {'Conf':>5} {'Pips':>6}  Reason")
        for r in wins_sorted:
            print(
                f"{r.alert_id:>5} {r.epic:<22} {r.signal_type:<5} "
                f"{(r.claude_score or 0):>5} {r.confidence:>5.2f} {r.pips:>+6.1f}  "
                f"{r.reason_snippet[:60]}"
            )

    # Top worst rejections claude got RIGHT on
    losses_sorted = sorted(losses, key=lambda r: r.pips)[:10]
    if losses_sorted:
        print(f"\n{'-' * 72}")
        print("TOP 10 'CLAUDE WAS RIGHT' (rejections that would have hit SL)")
        print(f"{'-' * 72}")
        print(f"{'ID':>5} {'Pair':<22} {'Type':<5} {'Score':>5} {'Conf':>5} {'Pips':>6}  Reason")
        for r in losses_sorted:
            print(
                f"{r.alert_id:>5} {r.epic:<22} {r.signal_type:<5} "
                f"{(r.claude_score or 0):>5} {r.confidence:>5.2f} {r.pips:>+6.1f}  "
                f"{r.reason_snippet[:60]}"
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
                   LEFT(COALESCE(claude_reason, ''), 120) AS reason_snippet
            FROM alert_history
            WHERE claude_decision = 'REJECT'
              AND price IS NOT NULL
            ORDER BY alert_timestamp ASC
            """
        )
        rows = cur.fetchall()

        results: list[Result] = []
        for row in rows:
            epic = row["epic"]
            sl_pips, tp_pips = pair_cfg.get(epic, (DEFAULT_SL_PIPS, DEFAULT_TP_PIPS))
            row["sl_pips"] = sl_pips
            row["tp_pips"] = tp_pips
            results.append(simulate(cur, row))

    summarize(results)


if __name__ == "__main__":
    main()
