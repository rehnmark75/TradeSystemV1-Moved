"""
RANGE_FADE Diagnostic Pass 3 — Optimized

Two checks in ONE joint pass per trade (not separate passes):
1. For DOA: compute 72h unconditional MFE via numpy cummax on candle window arrays.
2. For timeout audit: evaluate SL=25 × {2.0, 2.5, 3.0} cells simultaneously.

Both checks run together per trade → single candle-load, single iteration.
NO DB ROWS MODIFIED.
"""

import sys
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

EXECUTION_IDS = {
    'CS.D.EURUSD.CEEM.IP': 7777,
    'CS.D.GBPUSD.MINI.IP': 7778,
    'CS.D.USDJPY.MINI.IP': 7779,
    'CS.D.USDCAD.MINI.IP': 7780,
    'CS.D.USDCHF.MINI.IP': 7781,
    'CS.D.EURJPY.MINI.IP': 7782,
    'CS.D.AUDJPY.MINI.IP': 7783,
    'CS.D.AUDUSD.MINI.IP': 7784,
    'CS.D.NZDUSD.MINI.IP': 7785,
}
JPY_EPICS = {'CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP'}
ENTRY_COST_PIPS = 2.0
HORIZON_BARS = 72 * 60   # 4320 bars at 1m

# Borderline cells to audit for timeout contamination
AUDIT_SL = 25
AUDIT_R_LIST = [2.0, 2.5, 3.0]


def get_pip_size(epic):
    return 0.01 if epic in JPY_EPICS else 0.0001


def get_db():
    return psycopg2.connect(host='postgres', database='forex', user='postgres', password='postgres')


def load_signals(conn):
    exec_ids = list(EXECUTION_IDS.values())
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT bs.epic, bs.signal_timestamp AS ts, bs.signal_type,
                   bs.entry_price::float AS raw_price
            FROM backtest_signals bs
            WHERE bs.execution_id = ANY(%s)
            ORDER BY bs.epic, bs.signal_timestamp
        """, (exec_ids,))
        rows = cur.fetchall()
    signals = pd.DataFrame(rows, columns=['epic', 'ts', 'signal_type', 'raw_price'])
    signals['ts'] = pd.to_datetime(signals['ts'])
    return signals


def load_candles(conn, epic):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT start_time,
                   open::float, high::float, low::float, close::float
            FROM ig_candles
            WHERE epic = %s AND timeframe = 1
              AND start_time >= %s AND start_time <= %s
            ORDER BY start_time ASC
        """, (epic, datetime(2025, 9, 18), datetime(2026, 4, 4)))
        rows = cur.fetchall()
    times = np.array([r[0] for r in rows], dtype='datetime64[s]')
    highs = np.array([r[2] for r in rows], dtype=np.float64)
    lows  = np.array([r[3] for r in rows], dtype=np.float64)
    closes= np.array([r[4] for r in rows], dtype=np.float64)
    return times, highs, lows, closes


def run():
    log.info("=" * 70)
    log.info("RANGE_FADE DIAGNOSTIC PASS 3 — SINGLE JOINT PASS")
    log.info("NO DB ROWS MODIFIED")
    log.info("=" * 70)

    conn = get_db()
    signals = load_signals(conn)
    log.info(f"Loaded {len(signals)} signals")

    log.info("Loading 1m candles per pair...")
    candle_cache = {}
    for epic in EXECUTION_IDS:
        candle_cache[epic] = load_candles(conn, epic)
        t, h, l, c = candle_cache[epic]
        log.info(f"  {epic.split('.')[-2]}: {len(t)} candles")
    conn.close()

    # Per-trade results storage
    # DOA analysis (losers at SL=8)
    loser_mfe_72h = []
    n_sl8_winners = 0
    n_sl8_losers = 0
    n_sl8_timeouts = 0

    # Audit cells: SL=25 × R={2.0, 2.5, 3.0}
    # For each cell: separate lists for clean (WIN/LOSS) and timeout outcomes
    audit_pips_clean = {r: [] for r in AUDIT_R_LIST}
    audit_pips_to    = {r: [] for r in AUDIT_R_LIST}   # timeouts
    audit_n_to       = {r: 0 for r in AUDIT_R_LIST}
    audit_total      = {r: 0 for r in AUDIT_R_LIST}

    n_total = len(signals)
    log.info(f"\nRunning joint pass over {n_total} trades...")

    for idx, row in signals.iterrows():
        epic        = row['epic']
        ts          = row['ts']
        signal_type = row['signal_type']
        raw_price   = float(row['raw_price'])
        pip_size    = get_pip_size(epic)
        side        = 'BUY' if signal_type == 'BULL' else 'SELL'

        fill = (raw_price + ENTRY_COST_PIPS * pip_size if side == 'BUY'
                else raw_price - ENTRY_COST_PIPS * pip_size)

        times, highs, lows, closes = candle_cache[epic]

        # Find start index (first bar strictly after entry_ts)
        ts_np = np.datetime64(ts.to_pydatetime().replace(tzinfo=None), 's')
        start_idx = np.searchsorted(times, ts_np, side='right')
        end_idx = min(start_idx + HORIZON_BARS, len(times))

        if start_idx >= len(times) or start_idx >= end_idx:
            # No data — treat as timeout at 0
            n_sl8_timeouts += 1
            for r in AUDIT_R_LIST:
                audit_n_to[r] += 1
                audit_pips_to[r].append(0.0)
                audit_total[r] += 1
            continue

        h_win = highs[start_idx:end_idx]
        l_win = lows[start_idx:end_idx]
        c_win = closes[start_idx:end_idx]

        # -----------------------------------------------------------------------
        # STEP A: Unconditional 72h MFE (no early exit)
        # Favorable excursion per bar (relative to fill)
        # -----------------------------------------------------------------------
        if side == 'BUY':
            fav_per_bar = (h_win - fill) / pip_size   # how far price went in our favour
            adv_per_bar = (fill - l_win) / pip_size   # how far price went against us
        else:
            fav_per_bar = (fill - l_win) / pip_size
            adv_per_bar = (h_win - fill) / pip_size

        mfe_72h = float(np.max(np.maximum(fav_per_bar, 0.0)))

        # -----------------------------------------------------------------------
        # STEP B: First-passage at SL=8, R=2.0 (TP=16)
        # -----------------------------------------------------------------------
        sl_pips = 8
        tp_pips = sl_pips * 2.0
        sl_price = fill - sl_pips * pip_size if side == 'BUY' else fill + sl_pips * pip_size
        tp_price = fill + tp_pips * pip_size if side == 'BUY' else fill - tp_pips * pip_size

        outcome_sl8 = None
        pips_sl8    = 0.0
        for i in range(len(h_win)):
            h, l = h_win[i], l_win[i]
            if side == 'BUY':
                hit_sl = l <= sl_price
                hit_tp = h >= tp_price
            else:
                hit_sl = h >= sl_price
                hit_tp = l <= tp_price
            if hit_tp and hit_sl:
                outcome_sl8 = 'LOSS'
                pips_sl8 = -sl_pips
                break
            elif hit_tp:
                outcome_sl8 = 'WIN'
                pips_sl8 = tp_pips
                break
            elif hit_sl:
                outcome_sl8 = 'LOSS'
                pips_sl8 = -sl_pips
                break

        if outcome_sl8 is None:
            outcome_sl8 = 'TIMEOUT'
            pips_sl8 = (c_win[-1] - fill) / pip_size if side == 'BUY' else (fill - c_win[-1]) / pip_size
            n_sl8_timeouts += 1
        elif outcome_sl8 == 'WIN':
            n_sl8_winners += 1
        else:
            n_sl8_losers += 1
            # Store 72h MFE for losers (DOA analysis)
            loser_mfe_72h.append(mfe_72h)

        # -----------------------------------------------------------------------
        # STEP C: Audit cells SL=25 × R={2.0, 2.5, 3.0}
        # -----------------------------------------------------------------------
        for r_mult in AUDIT_R_LIST:
            audit_sl = 25
            audit_tp = audit_sl * r_mult
            sl_a = fill - audit_sl * pip_size if side == 'BUY' else fill + audit_sl * pip_size
            tp_a = fill + audit_tp * pip_size if side == 'BUY' else fill - audit_tp * pip_size

            out_a = None
            pips_a = 0.0
            for i in range(len(h_win)):
                h, l = h_win[i], l_win[i]
                if side == 'BUY':
                    hit_sl = l <= sl_a
                    hit_tp = h >= tp_a
                else:
                    hit_sl = h >= sl_a
                    hit_tp = l <= tp_a
                if hit_tp and hit_sl:
                    out_a = 'LOSS'
                    pips_a = -float(audit_sl)
                    break
                elif hit_tp:
                    out_a = 'WIN'
                    pips_a = float(audit_tp)
                    break
                elif hit_sl:
                    out_a = 'LOSS'
                    pips_a = -float(audit_sl)
                    break

            if out_a is None:
                # Timeout — mark to market
                timeout_p = (c_win[-1] - fill) / pip_size if side == 'BUY' else (fill - c_win[-1]) / pip_size
                audit_pips_to[r_mult].append(timeout_p)
                audit_n_to[r_mult] += 1
            else:
                audit_pips_clean[r_mult].append(pips_a)

            audit_total[r_mult] += 1

        if (idx + 1) % 500 == 0:
            log.info(f"  Processed {idx + 1}/{n_total}...")

    log.info(f"  Done. SL=8: {n_sl8_winners} W / {n_sl8_losers} L / {n_sl8_timeouts} TO")

    # -----------------------------------------------------------------------
    # RESULTS: DOA
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("CHECK 1: DIE-ON-ARRIVAL — 72h unconditional MFE for SL=8 LOSERS")
    log.info("=" * 70)

    if loser_mfe_72h:
        arr = np.array(loser_mfe_72h)
        n_losers = len(arr)

        doa  = np.sum(arr < 2) / n_losers
        r8   = np.sum(arr >= 8)  / n_losers
        r16  = np.sum(arr >= 16) / n_losers
        r25  = np.sum(arr >= 25) / n_losers
        r50  = np.sum(arr >= 50) / n_losers
        r62  = np.sum(arr >= 62) / n_losers
        r75  = np.sum(arr >= 75) / n_losers

        log.info(f"\n  Losers (stopped at SL=8): n={n_losers}")
        log.info(f"  Die-on-arrival MFE < 2 pips in 72h: {100*doa:.1f}%")
        log.info(f"  (The spec says 'dominant' = this fraction is significant, not a hard 50% line)")
        log.info(f"")
        log.info(f"  Rescuable fractions — fraction of losers that COULD have been saved")
        log.info(f"  by a wider TP (IF the SL had also been widened to avoid early stop):")
        log.info(f"    MFE >= 8 pips  (SL=8 doesn't help, entry already lost):  {100*r8:.1f}%")
        log.info(f"    MFE >= 16 pips (TP=16 via R=2 at SL=8):                  {100*r16:.1f}%")
        log.info(f"    MFE >= 25 pips (needs SL >= 25 to survive):               {100*r25:.1f}%")
        log.info(f"    MFE >= 50 pips (TP=50 = SL=25×R=2):                       {100*r50:.1f}%")
        log.info(f"    MFE >= 62 pips (TP=62 = SL=25×R=2.5):                     {100*r62:.1f}%")
        log.info(f"    MFE >= 75 pips (TP=75 = SL=25×R=3):                       {100*r75:.1f}%")
        log.info(f"")
        log.info(f"  Percentiles: P10={np.percentile(arr,10):.1f} / P25={np.percentile(arr,25):.1f} / "
                 f"P50={np.percentile(arr,50):.1f} / P75={np.percentile(arr,75):.1f} / P90={np.percentile(arr,90):.1f} pips")
        log.info(f"  Mean: {np.mean(arr):.2f} pips  Median: {np.median(arr):.2f} pips")

        if doa > 0.50:
            log.info(f"\n  DOA RULING: DOMINANT (>50%) => OR-branch fires => NO-GO")
        elif r75 < 0.10:
            log.info(f"\n  DOA RULING: <50% but only {100*r75:.1f}% reach 75 pip TP => width cannot rescue meaningful volume")
        else:
            log.info(f"\n  DOA RULING: <50% die-on-arrival; {100*r75:.1f}% reach TP=75 in 72h")

    # -----------------------------------------------------------------------
    # RESULTS: Timeout-excluded PF
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("CHECK 2: TIMEOUT-EXCLUDED PF for SL=25 borderline cells")
    log.info("=" * 70)

    for r_mult in AUDIT_R_LIST:
        tp_pips = AUDIT_SL * r_mult
        pips_c  = audit_pips_clean[r_mult]
        pips_to = audit_pips_to[r_mult]
        n_to    = audit_n_to[r_mult]
        n_all   = audit_total[r_mult]
        n_clean = len(pips_c)

        all_pips = pips_c + pips_to
        wins_all = sum(1 for p in all_pips if p > 0)
        wins_cln = sum(1 for p in pips_c if p > 0)

        def pf(pips_list):
            wins   = sum(p for p in pips_list if p > 0)
            losses = sum(abs(p) for p in pips_list if p < 0)
            return wins / losses if losses > 0 else float('inf')

        pf_full  = pf(all_pips)
        pf_clean = pf(pips_c) if pips_c else float('nan')
        wr_clean = wins_cln / n_clean if n_clean > 0 else 0.0

        # Geometric expected WR if cleanly resolved at this R-multiple
        # At PF=1.2 with pure WIN/LOSS: WR * R / (1-WR) = 1.2 => WR = 1.2 / (R + 1.2)
        breakeven_wr = 1.2 / (r_mult + 1.2)

        log.info(f"\n  SL=25, R={r_mult:.1f}, TP={tp_pips:.0f} pips:")
        log.info(f"    Total={n_all}, Timeouts={n_to} ({100*n_to/n_all:.1f}%), Cleanly resolved={n_clean}")
        log.info(f"    PF (incl. timeouts, mark-to-market): {pf_full:.3f}")
        log.info(f"    PF (cleanly resolved only):          {pf_clean:.3f}  WR={100*wr_clean:.1f}%")
        log.info(f"    Needed WR for PF=1.2 at R={r_mult:.1f}:   {100*breakeven_wr:.1f}%")

        survived = pf_clean >= 1.2 if not np.isnan(pf_clean) else False
        log.info(f"    SURVIVES timeout exclusion: {'YES' if survived else 'NO'}")

    # -----------------------------------------------------------------------
    # FINAL VERDICT
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("COMBINED VERDICT")
    log.info("=" * 70)

    if loser_mfe_72h:
        arr = np.array(loser_mfe_72h)
        doa_pct = 100 * np.sum(arr < 2) / len(arr)
        r75_pct = 100 * np.sum(arr >= 75) / len(arr)

    any_cell_survives = any(
        (lambda p: sum(x for x in p if x > 0) / sum(abs(x) for x in p if x < 0) >= 1.2
         if any(x < 0 for x in p) else False)(audit_pips_clean[r])
        for r in AUDIT_R_LIST
    )

    log.info(f"""
  Locked decision rule (OR-structured):
    NO-GO if: (a) no robust contiguous region passes both halves, OR
              (b) losers die on arrival

  Check A (region, from sweep pass 1):
    SL=25 × R=2.5 and R=3.0 both passed H1/H2 with pooled PF >= 1.2
    => Region check: PASSED (2 adjacent cells)

  Check B (DOA, from this pass):
    {doa_pct:.1f}% of SL=8 losers have MFE < 2 pips in the full 72h window
    {r75_pct:.1f}% of SL=8 losers could have reached TP=75 in 72h

  Check C (timeout contamination, from this pass):
    PF of passing cells when timeouts are excluded = see above
    any_cell_survives_clean: {any_cell_survives}
    """)

    if not any_cell_survives:
        log.info("  FINAL VERDICT: NO-GO")
        log.info("  Reason: Passing cells do not survive timeout exclusion —")
        log.info("  the apparent edge at SL=25 wide brackets is driven by 72h")
        log.info("  mark-to-market residuals, not genuine TP-reaching exits.")
        log.info("  DO NOT widen SL/TP for RANGE_FADE based on this sweep.")
    elif doa_pct > 50:
        log.info("  FINAL VERDICT: NO-GO")
        log.info("  Reason: Die-on-arrival fraction > 50% — width cannot rescue the majority of losers.")
    else:
        log.info("  FINAL VERDICT: GO (contingent)")
        log.info("  Both the region test and timeout-exclusion survived.")
        log.info("  NOTE: DOA fraction is near 50%; proceed with extreme caution.")

    log.info("\n=== COMPLETE. NO DB ROWS WERE MODIFIED. ===")


if __name__ == '__main__':
    run()
