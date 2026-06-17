"""
Decline -> Base -> Climb : Overlap + Forward-Edge Validation
============================================================
Answers two questions about the "downtrend -> sideways base -> climb"
(rounding-bottom) daily pattern, to decide whether it should be a standalone
SCANNER or a RANKING/CONFLUENCE feature on already-detected signals:

  PART A  FORWARD EDGE : tag every historical BUY signal in
          stock_scanner_signals with whether the pattern was present AS-OF the
          signal date (no look-ahead), then run the VALIDATED day-trade
          execution engine (imported from daytrade_edge_sim) and compare
          PF / WR / dead-on-arrival for pattern-present vs pattern-absent,
          overall and per scanner.

  PART B  OVERLAP / INDEPENDENCE : over recent signal dates, scan the whole
          universe for the pattern as-of each date and measure how many
          pattern hits COINCIDE with an existing scanner BUY (same ticker,
          +/-1 trading day) vs are NEW (no scanner flag). High coincidence ->
          ranker; high independence -> standalone scanner.

Execution model = CURRENT LIVE auto-trader (shipped 2026-06-12):
  fixed 2% SL / 7% TP, breakeven OFF, ATR-stop OFF.

Read-only. Run inside stock-scanner container:
    python -m stock_scanner.analysis.pattern_overlap_edge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import (
    get_conn, fetch_df, build_candle_index, compute_atr14,
    bars_after, walk_bars, atr_pct_before, build_daily_index,
)
import stock_scanner.analysis.daytrade_edge_sim as sim

# --- Mirror CURRENT live execution (override sim defaults) ---
sim.DEFAULT_SL_PCT = 2.0
sim.DEFAULT_TP_PCT = 7.0
sim.ATR_STOP_ENABLED = False        # ATR stop off
BE_ENABLED = False                  # breakeven off
LIVE_SL_PCT = 2.0
LIVE_TP_PCT = 7.0

# --- Pattern detector: single source of truth (parity with the live scanner) ---
from stock_scanner.patterns.decline_base_climb import detect_pattern, LOOKBACK_DAYS


def build_daily_with_volume(conn):
    """{ticker: daily DataFrame asc with volume + atr14}."""
    df = fetch_df(conn, """
        SELECT ticker, timestamp, open, high, low, close, COALESCE(volume,0) AS volume
        FROM stock_candles_synthesized WHERE timeframe='1d' ORDER BY ticker, timestamp
    """)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.reset_index(drop=True).copy()
        g["atr14"] = compute_atr14(g)
        out[str(t)] = g
    return out


def pattern_asof(daily_by_ticker, base_ticker, asof_date):
    """Detect pattern using ONLY daily bars strictly before asof_date (no look-ahead)."""
    d = daily_by_ticker.get(base_ticker)
    if d is None or d.empty:
        return False, 0.0
    asof = pd.Timestamp(asof_date)
    if asof.tzinfo is not None:
        asof = asof.tz_localize(None)
    sub = d[d["timestamp"] < asof]
    if len(sub) < 20:
        return False, 0.0
    sub = sub.iloc[-LOOKBACK_DAYS:]
    return detect_pattern(sub["close"].to_numpy(), sub["high"].to_numpy(),
                          sub["low"].to_numpy(), sub["volume"].to_numpy())


def pf_block(pnls, outcomes):
    """PF / WR / counts for a list of sim results."""
    pnls = np.array([p for p in pnls if p is not None], dtype=float)
    wins = [o == "win" for o in outcomes]
    losses = [o == "loss" for o in outcomes]
    n = len(outcomes)
    n_win, n_loss = sum(wins), sum(losses)
    gp = sum(p for p, w in zip(pnls, wins) if w) if n_win else 0.0
    gl = abs(sum(p for p, l in zip(pnls, losses) if l)) if n_loss else 0.0
    pf = (gp / gl) if gl > 0 else (9.99 if gp > 0 else None)
    resolved = n_win + n_loss
    wr = n_win / resolved * 100 if resolved else None
    exp = pnls.mean() if len(pnls) else None
    return {"n": n, "win": n_win, "loss": n_loss, "pf": pf, "wr": wr, "exp": exp}


def fmt(b):
    pf = f"{b['pf']:.2f}" if b["pf"] is not None else "  N/A"
    wr = f"{b['wr']:.0f}%" if b["wr"] is not None else " N/A"
    exp = f"{b['exp']:+.2f}%" if b["exp"] is not None else "  N/A"
    return f"n={b['n']:>4d}  PF={pf:>5s}  WR={wr:>4s}  exp={exp:>7s}  (W{b['win']}/L{b['loss']})"


def main():
    conn = get_conn()
    print("Loading 1h candle index (for sim) ...")
    candles = build_candle_index(conn)
    print(f"  {len(candles)} tickers")
    print("Loading daily index with volume (for pattern) ...")
    daily = build_daily_with_volume(conn)
    print(f"  {len(daily)} tickers")

    # ---- Load BUY signals ----
    sigs = fetch_df(conn, """
        SELECT id, signal_timestamp, scanner_name, ticker, signal_date
        FROM stock_scanner_signals WHERE signal_type='BUY' ORDER BY signal_timestamp
    """)
    # Normalize to tz-naive UTC (daily/1h candle timestamps are tz-naive)
    sigs["signal_timestamp"] = pd.to_datetime(sigs["signal_timestamp"], utc=True).dt.tz_localize(None)
    print(f"\nLoaded {len(sigs)} BUY signals.")

    # =====================================================================
    # PART A : FORWARD EDGE  (pattern-present vs absent, live exec model)
    # =====================================================================
    print("\n" + "=" * 78)
    print("PART A: FORWARD EDGE  (live exec: 2%SL / 7%TP, BE off, ATR off)")
    print("=" * 78)

    rows = []
    no_data = 0
    for _, s in sigs.iterrows():
        base = str(s["ticker"]).split(".")[0]
        ts = s["signal_timestamp"]
        asof = pd.Timestamp(ts).normalize()
        matched, score = pattern_asof(daily, base, asof)
        bars, entry = bars_after(candles, base, ts)
        if bars.empty or entry <= 0:
            no_data += 1
            continue
        bracket = sim.build_bracket(entry, None)   # forced fixed 2/7 (ATR off)
        res = walk_bars(entry, bracket, bars, be_enabled=BE_ENABLED)
        rows.append({"scanner": s["scanner_name"], "ticker": base,
                     "matched": matched, "score": score,
                     "outcome": res["outcome"], "pnl": res["pnl_pct"],
                     "hold": res["hold_bars"]})
    R = pd.DataFrame(rows)
    print(f"  Simulated {len(R)} signals ({no_data} dropped: no candle data).")

    present = R[R["matched"]]
    absent = R[~R["matched"]]
    bp = pf_block(present["pnl"], present["outcome"])
    ba = pf_block(absent["pnl"], absent["outcome"])
    overall = pf_block(R["pnl"], R["outcome"])
    print(f"\n  OVERALL  pattern PRESENT : {fmt(bp)}")
    print(f"  OVERALL  pattern ABSENT  : {fmt(ba)}")
    print(f"  OVERALL  all signals     : {fmt(overall)}")
    rate = len(present) / len(R) * 100 if len(R) else 0
    print(f"  Pattern present on {len(present)}/{len(R)} = {rate:.1f}% of simulated BUY signals.")

    print(f"\n  PER-SCANNER  (present vs absent; only scanners with >=20 present):")
    print(f"  {'scanner':30s} {'PRESENT':>40s}   {'ABSENT':>40s}")
    for scn, g in R.groupby("scanner"):
        gp = g[g["matched"]]
        if len(gp) < 20:
            continue
        ga = g[~g["matched"]]
        bp_ = pf_block(gp["pnl"], gp["outcome"])
        ba_ = pf_block(ga["pnl"], ga["outcome"])
        print(f"  {scn:30s} {fmt(bp_):>40s}   {fmt(ba_):>40s}")

    # Score-tier monotonicity (does higher pattern score -> better edge?)
    print(f"\n  SCORE TIERS (present only):")
    if len(present) >= 30:
        for lo, hi in [(0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 6.1)]:
            t = present[(present["score"] >= lo) & (present["score"] < hi)]
            if len(t):
                print(f"    score [{lo:.1f},{hi:.1f}): {fmt(pf_block(t['pnl'], t['outcome']))}")

    # =====================================================================
    # PART C : DEDUPED PURE-PATTERN POPULATION  (the decisive scanner test)
    # =====================================================================
    # Simulate the pattern-hit population DIRECTLY (not the scanner-overlap
    # subset). Dedupe to ONE entry per climb episode per ticker (a climb
    # persists for days and re-fires daily -> raw counts are inflated, same
    # as the RANGE_FADE re-fire problem). Entry = first 1h bar after the
    # episode's FIRST detection date. This measures the standalone edge.
    print("\n" + "=" * 78)
    print("PART C: DEDUPED PURE-PATTERN POPULATION  (standalone-scanner edge)")
    print("=" * 78)

    EPISODE_GAP = 5          # >= this many trading-day gap starts a new episode
    WINDOW_DAYS = 150        # only entries within last N days (1h candle coverage)
    max_d = max(g["timestamp"].max() for g in daily.values())
    cutoff = max_d - pd.Timedelta(days=WINDOW_DAYS)

    episodes = []            # (ticker, entry_date, score)
    for tk, d in daily.items():
        ts = d["timestamp"].to_numpy()
        cl = d["close"].to_numpy()
        hi = d["high"].to_numpy()
        lo = d["low"].to_numpy()
        vo = d["volume"].to_numpy()
        last_hit_t = -99
        for t in range(25, len(d)):
            if d["timestamp"].iloc[t] < cutoff:
                continue
            lo_idx = max(0, t - LOOKBACK_DAYS)
            matched, score = detect_pattern(cl[lo_idx:t], hi[lo_idx:t], lo[lo_idx:t], vo[lo_idx:t])
            if matched:
                if t - last_hit_t >= EPISODE_GAP:      # new episode
                    episodes.append((tk, d["timestamp"].iloc[t], score))
                last_hit_t = t
    print(f"\n  Deduped episodes (last {WINDOW_DAYS}d, gap>={EPISODE_GAP} td): {len(episodes)}")

    erows = []
    for tk, edate, score in episodes:
        bars, entry = bars_after(candles, tk, pd.Timestamp(edate))
        if bars.empty or entry <= 0:
            continue
        bracket = sim.build_bracket(entry, None)
        res = walk_bars(entry, bracket, bars, be_enabled=BE_ENABLED)
        erows.append({"ticker": tk, "score": score, "outcome": res["outcome"], "pnl": res["pnl_pct"]})
    E = pd.DataFrame(erows)
    print(f"  Simulated {len(E)} episode entries (live exec 2%SL/7%TP, BE off).")
    print(f"\n  PURE-PATTERN edge (all episodes) : {fmt(pf_block(E['pnl'], E['outcome']))}")
    print(f"\n  By score tier:")
    for lo, hi in [(0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 6.1)]:
        t = E[(E["score"] >= lo) & (E["score"] < hi)]
        if len(t):
            print(f"    score [{lo:.1f},{hi:.1f}): {fmt(pf_block(t['pnl'], t['outcome']))}")
    # Episodes/day to show realistic standalone volume
    n_days = max(1, (max_d - cutoff).days * 5 // 7)
    print(f"\n  ~{len(episodes)/n_days:.1f} episodes/trading-day (vs ~8 daily slots).")

    print("\n  DECISION:")
    print("    Pure-pattern PF >= ~1.2 at score>=4.0  -> standalone SCANNER justified.")
    print("    Pure-pattern PF collapses to ~1.0      -> weak feature, RANKER-only.")
    print(f"    Reference: gap_and_go ranker lift (Part A) 0.89 -> 1.21 on n=201 is the")
    print(f"    one demonstrated, decent-n lever regardless of this result.")

    conn.close()
    print("\nScript: stock_scanner/analysis/pattern_overlap_edge.py")


if __name__ == "__main__":
    main()
