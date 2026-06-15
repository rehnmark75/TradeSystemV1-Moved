"""
RANGE_FADE Wider SL/TP — Fixed-Bracket Sweep
Pre-registered experiment: RANGE_FADE_WIDTH_SWEEP.md

EXIT-ONLY on a FROZEN entry set. No DB rows modified.
Decision rule is LOCKED before looking.

Entry set: 1,946 RANGE_FADE signals from honest OOS window
  Sep 18 2025 → Mar 31 2026, sourced from backtest_signals
  in executions 7777-7785 (one per pair, confirmed counts).

Data source: ig_candles (timeframe=1, live 1m bars).

Author: quantitative-researcher agent
Date: 2026-06-15
"""

import sys
import os
import logging
from datetime import datetime, timedelta, timezone
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Execution IDs for the frozen OOS backtest set (Jun 14 2026 runs)
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

# Mandated integrity counts from spec
EXPECTED_COUNTS = {
    'CS.D.EURUSD.CEEM.IP': 189,
    'CS.D.GBPUSD.MINI.IP': 105,
    'CS.D.USDJPY.MINI.IP': 153,
    'CS.D.USDCAD.MINI.IP': 414,
    'CS.D.USDCHF.MINI.IP': 397,
    'CS.D.EURJPY.MINI.IP': 12,
    'CS.D.AUDJPY.MINI.IP': 40,
    'CS.D.AUDUSD.MINI.IP': 376,
    'CS.D.NZDUSD.MINI.IP': 260,
}

PAIR_NAMES = {
    'CS.D.EURUSD.CEEM.IP': 'EURUSD',
    'CS.D.GBPUSD.MINI.IP': 'GBPUSD',
    'CS.D.USDJPY.MINI.IP': 'USDJPY',
    'CS.D.USDCAD.MINI.IP': 'USDCAD',
    'CS.D.USDCHF.MINI.IP': 'USDCHF',
    'CS.D.EURJPY.MINI.IP': 'EURJPY',
    'CS.D.AUDJPY.MINI.IP': 'AUDJPY',
    'CS.D.AUDUSD.MINI.IP': 'AUDUSD',
    'CS.D.NZDUSD.MINI.IP': 'NZDUSD',
}

# JPY pairs use 0.01 per pip; all others 0.0001 per pip
JPY_EPICS = {'CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP'}

# Entry cost per spec: 2.0 pips worsening the fill
ENTRY_COST_PIPS = 2.0

# Grid definition
SL_VALUES = [8, 12, 16, 20, 25, 30]
R_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0]

# Horizon cap
HORIZON_HOURS = 72

# Robustness thresholds
MIN_PF = 1.2
MIN_N_HALF = 100

# H1/H2 split
H1_START = datetime(2025, 9, 18, 0, 0, 0)
H1_END = datetime(2025, 12, 31, 23, 59, 59)
H2_START = datetime(2026, 1, 1, 0, 0, 0)
H2_END = datetime(2026, 3, 31, 23, 59, 59)


def get_db_connection():
    return psycopg2.connect(
        host='postgres',
        database='forex',
        user='postgres',
        password='postgres',
    )


def get_pip_size(epic: str) -> float:
    return 0.01 if epic in JPY_EPICS else 0.0001


def load_signals(conn) -> pd.DataFrame:
    """Load the frozen entry set from backtest_signals for the OOS executions."""
    exec_ids = list(EXECUTION_IDS.values())
    query = """
        SELECT
            bs.epic,
            bs.signal_timestamp AS ts,
            bs.signal_type,
            bs.entry_price::float AS raw_price
        FROM backtest_signals bs
        WHERE bs.execution_id = ANY(%s)
        ORDER BY bs.epic, bs.signal_timestamp
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, (exec_ids,))
        rows = cur.fetchall()

    signals = pd.DataFrame(rows, columns=['epic', 'ts', 'signal_type', 'raw_price'])
    signals['ts'] = pd.to_datetime(signals['ts'])
    return signals


def verify_integrity(signals: pd.DataFrame) -> bool:
    """Verify signal counts match spec mandated totals."""
    ok = True
    log.info("=== INTEGRITY CHECK ===")
    total = 0
    for epic, expected in EXPECTED_COUNTS.items():
        subset = signals[signals['epic'] == epic]
        actual = len(subset)
        total += actual
        status = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        pair = PAIR_NAMES[epic]
        log.info(f"  {pair}: {actual} signals [{status}]")
        if actual != expected:
            ok = False
    log.info(f"  TOTAL: {total} signals (expected 1946)")
    if total != 1946:
        log.error(f"TOTAL MISMATCH: got {total}, expected 1946")
        ok = False
    return ok


def load_candles_for_epic(conn, epic: str) -> pd.DataFrame:
    """
    Bulk-load ALL 1m candles for this epic over the OOS window + 72h buffer.
    Window: Sep 18 2025 → Apr 4 2026 (=Mar 31 + 72h extra buffer for late entries).
    """
    query = """
        SELECT start_time, open, high, low, close
        FROM ig_candles
        WHERE epic = %s
          AND timeframe = 1
          AND start_time >= %s
          AND start_time <= %s
        ORDER BY start_time ASC
    """
    start = datetime(2025, 9, 18, 0, 0, 0)
    end = datetime(2026, 4, 4, 0, 0, 0)  # Apr 4 covers all 72h horizons

    with conn.cursor() as cur:
        cur.execute(query, (epic, start, end))
        rows = cur.fetchall()

    if not rows:
        log.warning(f"No candles found for {epic}")
        return pd.DataFrame(columns=['start_time', 'open', 'high', 'low', 'close'])

    df = pd.DataFrame(rows, columns=['start_time', 'open', 'high', 'low', 'close'])
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.set_index('start_time').sort_index()
    log.info(f"  Loaded {len(df):,} 1m candles for {PAIR_NAMES.get(epic, epic)}: "
             f"{df.index[0]} → {df.index[-1]}")
    return df


def compute_first_passage(
    candles: pd.DataFrame,
    entry_ts: datetime,
    fill_price: float,
    side: str,    # 'BUY' or 'SELL'
    sl_price: float,
    tp_price: float,
    pip_size: float,
) -> dict:
    """
    First-passage resolution on 1m candles starting from the bar AFTER entry_ts.
    Pessimistic straddle: if both SL and TP hit within same 1m bar, counts as LOSS.
    72h timeout: close at last available price.

    Returns dict with:
        outcome: 'WIN', 'LOSS', 'TIMEOUT'
        pips: pip P&L (positive = win)
        bars_to_resolution: integer
        mfe_pips: max favorable excursion before resolution
        mae_pips: max adverse excursion before resolution (positive = bad)
    """
    # Start scanning from the bar AFTER the signal bar
    scan_start = entry_ts + timedelta(minutes=1)
    scan_end = entry_ts + timedelta(hours=HORIZON_HOURS)

    # Slice candles in-memory
    window = candles.loc[scan_start:scan_end] if not candles.empty else pd.DataFrame()

    if window.empty:
        # No data after entry → timeout at fill price (no movement)
        return {
            'outcome': 'TIMEOUT',
            'pips': 0.0,
            'bars_to_resolution': 0,
            'mfe_pips': 0.0,
            'mae_pips': 0.0,
        }

    mfe_pips = 0.0
    mae_pips = 0.0

    for i, (bar_ts, row) in enumerate(window.iterrows()):
        h = float(row['high'])
        l = float(row['low'])

        if side == 'BUY':
            favorable = h - fill_price
            adverse = fill_price - l
            hit_sl = l <= sl_price
            hit_tp = h >= tp_price
        else:  # SELL
            favorable = fill_price - l
            adverse = h - fill_price
            hit_sl = h >= sl_price
            hit_tp = l <= tp_price

        # Update MFE/MAE (in pips)
        fav_pips = max(0.0, favorable / pip_size)
        adv_pips = max(0.0, adverse / pip_size)
        mfe_pips = max(mfe_pips, fav_pips)
        mae_pips = max(mae_pips, adv_pips)

        if hit_tp and hit_sl:
            # Pessimistic straddle → LOSS
            loss_pips = (fill_price - sl_price) / pip_size if side == 'BUY' else (sl_price - fill_price) / pip_size
            return {
                'outcome': 'LOSS',
                'pips': -abs(loss_pips),
                'bars_to_resolution': i + 1,
                'mfe_pips': mfe_pips,
                'mae_pips': mae_pips,
            }
        elif hit_tp:
            win_pips = (tp_price - fill_price) / pip_size if side == 'BUY' else (fill_price - tp_price) / pip_size
            return {
                'outcome': 'WIN',
                'pips': abs(win_pips),
                'bars_to_resolution': i + 1,
                'mfe_pips': mfe_pips,
                'mae_pips': mae_pips,
            }
        elif hit_sl:
            loss_pips = (fill_price - sl_price) / pip_size if side == 'BUY' else (sl_price - fill_price) / pip_size
            return {
                'outcome': 'LOSS',
                'pips': -abs(loss_pips),
                'bars_to_resolution': i + 1,
                'mfe_pips': mfe_pips,
                'mae_pips': mae_pips,
            }

    # 72h timeout: close at last available price
    last_close = float(window.iloc[-1]['close'])
    if side == 'BUY':
        timeout_pips = (last_close - fill_price) / pip_size
    else:
        timeout_pips = (fill_price - last_close) / pip_size

    return {
        'outcome': 'TIMEOUT',
        'pips': timeout_pips,
        'bars_to_resolution': len(window),
        'mfe_pips': mfe_pips,
        'mae_pips': mae_pips,
    }


def compute_pf(pips_list: list) -> float:
    """Profit factor = sum of wins / sum of losses (absolute)."""
    wins = sum(p for p in pips_list if p > 0)
    losses = sum(abs(p) for p in pips_list if p < 0)
    if losses == 0:
        return float('inf') if wins > 0 else float('nan')
    return wins / losses


def run_sweep():
    """Main sweep logic."""
    log.info("=" * 70)
    log.info("RANGE_FADE WIDTH SWEEP — pre-registered experiment")
    log.info("EXIT-ONLY on frozen entry set. NO DB ROWS MODIFIED.")
    log.info("=" * 70)

    conn = get_db_connection()

    # -----------------------------------------------------------------------
    # 1. Load frozen entry set
    # -----------------------------------------------------------------------
    log.info("\n[1] Loading frozen entry set from backtest_signals...")
    signals = load_signals(conn)
    log.info(f"    Loaded {len(signals)} total signals")

    if not verify_integrity(signals):
        log.error("INTEGRITY CHECK FAILED — aborting")
        sys.exit(1)
    log.info("    Integrity check PASSED (1946 signals, all per-pair counts match)")

    # -----------------------------------------------------------------------
    # 2. Load 1m candles per pair
    # -----------------------------------------------------------------------
    log.info("\n[2] Bulk-loading 1m candles per pair...")
    candle_cache = {}
    for epic in EXECUTION_IDS:
        candle_cache[epic] = load_candles_for_epic(conn, epic)

    conn.close()

    # -----------------------------------------------------------------------
    # 3. Compute first-passage for all trades × all grid cells
    # -----------------------------------------------------------------------
    log.info("\n[3] Computing first-passage for all trades × 30 grid cells...")
    log.info(f"    Grid: SL in {SL_VALUES}, R in {R_MULTIPLES}")
    log.info(f"    Entry cost: {ENTRY_COST_PIPS} pips worsening fill (BUY+2, SELL-2)")

    # We'll store per-trade results for each (sl, r) cell
    # Structure: results[(sl_pips, r_mult)] = list of dicts
    results = defaultdict(list)

    # Also store mechanism diagnostic data (computed at current SL=8 equivalent)
    diag_records = []  # list of dicts: {epic, side, raw_price, fill, ts, ...}

    n_total = len(signals)
    n_processed = 0
    n_timeout = defaultdict(int)

    for _, row in signals.iterrows():
        epic = row['epic']
        ts = row['ts']
        signal_type = row['signal_type']  # 'BULL' or 'BEAR'
        raw_price = float(row['raw_price'])
        pip_size = get_pip_size(epic)

        side = 'BUY' if signal_type == 'BULL' else 'SELL'

        # Apply 2-pip entry cost per spec
        if side == 'BUY':
            fill = raw_price + ENTRY_COST_PIPS * pip_size
        else:
            fill = raw_price - ENTRY_COST_PIPS * pip_size

        candles = candle_cache[epic]

        # Run first-passage for each grid cell
        # We cache per-trade the 72h MFE/MAE (computed once at the first SL cell)
        trade_mfe = None
        trade_mae = None

        # Compute the 72h MFE/MAE diagnostic once per trade (horizon = 72h unconditional)
        diag_result = compute_first_passage(
            candles, ts, fill, side,
            sl_price=(fill - 1000 * pip_size) if side == 'BUY' else (fill + 1000 * pip_size),  # never hit
            tp_price=(fill + 1000 * pip_size) if side == 'BUY' else (fill - 1000 * pip_size),  # never hit
            pip_size=pip_size,
        )
        trade_mfe = diag_result['mfe_pips']
        trade_mae = diag_result['mae_pips']
        diag_records.append({
            'epic': epic,
            'ts': ts,
            'side': side,
            'mfe_pips': trade_mfe,
            'mae_pips': trade_mae,
            'half': 'H1' if ts <= H1_END else 'H2',
        })

        for sl_pips in SL_VALUES:
            sl_price = (fill - sl_pips * pip_size) if side == 'BUY' else (fill + sl_pips * pip_size)

            for r_mult in R_MULTIPLES:
                tp_pips = sl_pips * r_mult
                tp_price = (fill + tp_pips * pip_size) if side == 'BUY' else (fill - tp_pips * pip_size)

                res = compute_first_passage(
                    candles, ts, fill, side,
                    sl_price, tp_price, pip_size
                )

                if res['outcome'] == 'TIMEOUT':
                    n_timeout[(sl_pips, r_mult)] += 1

                results[(sl_pips, r_mult)].append({
                    'epic': epic,
                    'ts': ts,
                    'side': side,
                    'outcome': res['outcome'],
                    'pips': res['pips'],
                    'mfe_pips': res['mfe_pips'],
                    'mae_pips': res['mae_pips'],
                    'half': 'H1' if ts <= H1_END else 'H2',
                })

        n_processed += 1
        if n_processed % 200 == 0:
            log.info(f"    Processed {n_processed}/{n_total} trades...")

    log.info(f"    Done. Processed {n_processed} trades × 30 grid cells.")

    # -----------------------------------------------------------------------
    # 4. Mechanism Diagnostic
    # -----------------------------------------------------------------------
    log.info("\n[4] MECHANISM DIAGNOSTIC")
    diag_df = pd.DataFrame(diag_records)

    # Use the SL=8 cell results for the current-bracket losers
    sl8_results = pd.DataFrame(results[(8, 1.0)])  # SL=8, R=1.0 (losers same regardless of R for loss)
    # Actually need losers for all R multiples at SL=8; use SL=8 baseline
    sl8_full = []
    for r in R_MULTIPLES:
        for rec in results[(8, r)]:
            if rec['outcome'] == 'LOSS':
                sl8_full.append(rec)
                break  # Only once per trade (SL hit regardless of R)
    # Better: get all SL=8/R=3 results (R doesn't matter for loss; loss = SL hit)
    sl8_trades = pd.DataFrame(results[(8, 3.0)])  # R=3 means TP was harder → same SL behavior
    # Actually per spec: "of trades that lose at current SL=8"
    # Use SL=8, R=2 as representative (matches the current SL=8, TP=15 ≈ R=1.875)
    sl8_all = pd.DataFrame(results[(8, 2.0)])

    losers_at_sl8 = sl8_all[sl8_all['outcome'] == 'LOSS']
    winners_at_sl8 = sl8_all[sl8_all['outcome'] == 'WIN']
    n_losers = len(losers_at_sl8)
    n_winners = len(winners_at_sl8)
    n_all_sl8 = len(sl8_all)

    log.info(f"\n  Current bracket SL=8, R=2.0 (closest to live 8/15):")
    log.info(f"    Total trades: {n_all_sl8}, Winners: {n_winners}, Losers: {n_losers}")
    if n_all_sl8 > 0:
        log.info(f"    Win rate: {100*n_winners/n_all_sl8:.1f}%")

    # MFE distribution for losers (did they have any favorable movement?)
    if n_losers > 0:
        mfe_values = losers_at_sl8['mfe_pips'].values
        die_on_arrival = np.sum(mfe_values < 2) / n_losers
        rescuable_8 = np.sum(mfe_values >= 8) / n_losers
        rescuable_12 = np.sum(mfe_values >= 12) / n_losers
        rescuable_16 = np.sum(mfe_values >= 16) / n_losers

        log.info(f"\n  Loser MFE distribution (n={n_losers}):")
        log.info(f"    Die-on-arrival (MFE < 2 pips): {100*die_on_arrival:.1f}% — {'NO-GO signal' if die_on_arrival > 0.5 else 'below 50%'}")
        log.info(f"    Reached MFE >= 8 pips before SL:  {100*rescuable_8:.1f}% (rescuable by wider stop)")
        log.info(f"    Reached MFE >= 12 pips before SL: {100*rescuable_12:.1f}%")
        log.info(f"    Reached MFE >= 16 pips before SL: {100*rescuable_16:.1f}%")
        log.info(f"    Median MFE:  {np.median(mfe_values):.1f} pips")
        log.info(f"    Mean MFE:    {np.mean(mfe_values):.1f} pips")

    # Winner left-on-table at TP=16 (SL=8×R=2)
    if n_winners > 0:
        winner_mfe = winners_at_sl8['mfe_pips'].values
        tp_pips_val = 16.0  # SL=8 × R=2
        left_on_table = winner_mfe - tp_pips_val
        log.info(f"\n  Winner left-on-table at TP={tp_pips_val:.0f} pips (SL=8, R=2):")
        log.info(f"    Mean left-on-table: {np.mean(left_on_table):.1f} pips")
        log.info(f"    Median left-on-table: {np.median(left_on_table):.1f} pips")
        log.info(f"    Winners with MFE > TP+5: {np.sum(winner_mfe > tp_pips_val+5)}/{n_winners} ({100*np.mean(winner_mfe > tp_pips_val+5):.0f}%)")

    # -----------------------------------------------------------------------
    # 5. Full PF Heatmap
    # -----------------------------------------------------------------------
    log.info("\n[5] POOLED PF HEATMAP (SL rows × R columns)")

    # Build summary table
    grid_data = {}
    for sl in SL_VALUES:
        for r in R_MULTIPLES:
            cell_records = pd.DataFrame(results[(sl, r)])
            n = len(cell_records)
            pips_list = cell_records['pips'].tolist()
            pf = compute_pf(pips_list)
            wins = sum(1 for p in pips_list if p > 0)
            wr = wins / n if n > 0 else 0.0
            exp = np.mean(pips_list) if pips_list else 0.0
            timeout_n = n_timeout[(sl, r)]

            # H1/H2 split
            h1 = cell_records[cell_records['half'] == 'H1']
            h2 = cell_records[cell_records['half'] == 'H2']
            h1_pf = compute_pf(h1['pips'].tolist())
            h2_pf = compute_pf(h2['pips'].tolist())
            h1_n = len(h1)
            h2_n = len(h2)

            robust = (
                pf >= MIN_PF
                and h1_pf >= MIN_PF and h1_n >= MIN_N_HALF
                and h2_pf >= MIN_PF and h2_n >= MIN_N_HALF
            )

            grid_data[(sl, r)] = {
                'n': n, 'pf': pf, 'wr': wr, 'exp': exp,
                'h1_n': h1_n, 'h1_pf': h1_pf,
                'h2_n': h2_n, 'h2_pf': h2_pf,
                'robust': robust,
                'timeout_n': timeout_n,
            }

    # Print PF heatmap
    log.info("\n  Pooled PF by (SL, R):")
    sl_r_label = 'SL\\R'
    header = f"{sl_r_label:>8}" + "".join(f"  R={r:.1f}" for r in R_MULTIPLES)
    log.info(header)
    log.info("-" * (8 + 10 * len(R_MULTIPLES)))
    for sl in SL_VALUES:
        row_str = f"  SL={sl:>2}"
        for r in R_MULTIPLES:
            d = grid_data[(sl, r)]
            pf_val = d['pf']
            mark = "*" if d['robust'] else " "
            if pf_val == float('inf'):
                cell = f"  {'inf':>5}{mark}"
            elif np.isnan(pf_val):
                cell = f"  {'nan':>5}{mark}"
            else:
                cell = f"  {pf_val:>5.2f}{mark}"
        row_str += "".join(
            (f"  {'inf':>5}{'*' if grid_data[(sl,r)]['robust'] else ' '}"
             if grid_data[(sl,r)]['pf'] == float('inf')
             else f"  {grid_data[(sl,r)]['pf']:>5.2f}{'*' if grid_data[(sl,r)]['robust'] else ' '}")
            for r in R_MULTIPLES
        )
        log.info(row_str)

    log.info("  (* = passes robustness: pooled PF>=1.2 AND H1 PF>=1.2 n>=100 AND H2 PF>=1.2 n>=100)")

    # Print n heatmap
    log.info("\n  N per cell (should be 1946 for all — timeout != loss of n):")
    log.info(f"{sl_r_label:>8}" + "".join(f"  R={r:.1f}" for r in R_MULTIPLES))
    log.info("-" * (8 + 10 * len(R_MULTIPLES)))
    for sl in SL_VALUES:
        row_str = f"  SL={sl:>2}" + "".join(f"  {grid_data[(sl,r)]['n']:>5} " for r in R_MULTIPLES)
        log.info(row_str)

    # -----------------------------------------------------------------------
    # 6. Robustness Detail for passing cells
    # -----------------------------------------------------------------------
    log.info("\n[6] ROBUSTNESS DETAIL (cells with pooled PF >= 1.2)")
    passing_cells = [(sl, r) for sl in SL_VALUES for r in R_MULTIPLES
                     if not np.isnan(grid_data[(sl,r)]['pf']) and grid_data[(sl,r)]['pf'] >= MIN_PF]

    if not passing_cells:
        log.info("  No cells achieve pooled PF >= 1.2.")
    else:
        log.info(f"  {len(passing_cells)} cells with PF >= 1.2:")
        for sl, r in passing_cells:
            d = grid_data[(sl, r)]
            tp = sl * r
            log.info(
                f"    SL={sl:>2} R={r:.1f} TP={tp:>4.0f}: PF={d['pf']:.3f} n={d['n']} WR={100*d['wr']:.1f}% exp={d['exp']:+.2f}pip "
                f"| H1: PF={d['h1_pf']:.3f} n={d['h1_n']} "
                f"| H2: PF={d['h2_pf']:.3f} n={d['h2_n']} "
                f"| Robust: {'YES *' if d['robust'] else 'NO'}"
            )

    # -----------------------------------------------------------------------
    # 7. Sanity anchor check at current bracket SL=8, R=1.875 (≈ 8/15)
    # -----------------------------------------------------------------------
    log.info("\n[7] SANITY ANCHOR: Current bracket SL=8, R≈2 (closest to live 8/15)")
    d8 = grid_data[(8, 2.0)]
    log.info(f"  SL=8, R=2.0 (TP=16): PF={d8['pf']:.3f}, WR={100*d8['wr']:.1f}%, n={d8['n']}")
    log.info(f"  Expected anchor: ~0.70 PF (from trailing-run pooled estimate)")
    if abs(d8['pf'] - 0.70) < 0.25:
        log.info(f"  Anchor check: PLAUSIBLE (within 0.25 of expected 0.70)")
    else:
        log.info(f"  Anchor check: DIVERGED — investigate parsing/first-passage before trusting grid")

    # -----------------------------------------------------------------------
    # 8. Contiguity check for LOCKED decision rule
    # -----------------------------------------------------------------------
    log.info("\n[8] CONTIGUITY CHECK FOR DECISION RULE")
    robust_cells = [(sl, r) for sl in SL_VALUES for r in R_MULTIPLES if grid_data[(sl,r)]['robust']]

    def are_adjacent(c1, c2):
        sl1, r1 = c1
        sl2, r2 = c2
        sl_idx1, sl_idx2 = SL_VALUES.index(sl1), SL_VALUES.index(sl2)
        r_idx1, r_idx2 = R_MULTIPLES.index(r1), R_MULTIPLES.index(r2)
        return (abs(sl_idx1 - sl_idx2) <= 1 and r_idx1 == r_idx2) or \
               (sl_idx1 == sl_idx2 and abs(r_idx1 - r_idx2) <= 1)

    def find_contiguous_regions(cells):
        if not cells:
            return []
        visited = set()
        regions = []
        for cell in cells:
            if cell in visited:
                continue
            region = [cell]
            visited.add(cell)
            queue = [cell]
            while queue:
                current = queue.pop(0)
                for other in cells:
                    if other not in visited and are_adjacent(current, other):
                        visited.add(other)
                        region.append(other)
                        queue.append(other)
            regions.append(region)
        return regions

    regions = find_contiguous_regions(robust_cells)

    if not robust_cells:
        log.info(f"  Robust cells (PF>=1.2, H1 PF>=1.2 n>=100, H2 PF>=1.2 n>=100): NONE")
    else:
        log.info(f"  Robust cells: {len(robust_cells)}")
        for sl, r in robust_cells:
            d = grid_data[(sl,r)]
            log.info(f"    SL={sl} R={r:.1f}: pooled PF={d['pf']:.3f}, H1 PF={d['h1_pf']:.3f} n={d['h1_n']}, H2 PF={d['h2_pf']:.3f} n={d['h2_n']}")

    largest_region = max(regions, key=len) if regions else []
    contiguous_pass = len(largest_region) >= 2 if regions else False

    # -----------------------------------------------------------------------
    # 9. VERDICT
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("VERDICT (LOCKED DECISION RULE)")
    log.info("=" * 70)

    # Check die-on-arrival first (if > 50% of losers die on arrival → additional NO-GO signal)
    doa_fraction = 0.0
    if n_losers > 0:
        mfe_values_losers = losers_at_sl8['mfe_pips'].values
        doa_fraction = np.sum(mfe_values_losers < 2) / n_losers

    if contiguous_pass:
        log.info(f"\n  VERDICT: GO")
        log.info(f"  Contiguous region of >= 2 robust cells found: {largest_region}")
        log.info(f"  Recommended bracket region identified above.")
        log.info(f"  NEXT STEP (separate): confirm under live trailing widened to match before any config change.")
    else:
        log.info(f"\n  VERDICT: NO-GO")
        log.info(f"  Reason: No contiguous region of >= 2 adjacent robust cells found.")
        if doa_fraction >= 0.5:
            log.info(f"  Additional: {100*doa_fraction:.0f}% of losers at SL=8 die-on-arrival (MFE < 2 pips).")
            log.info(f"  This confirms: entry edge is not realizable at any tested width.")
        else:
            log.info(f"  Die-on-arrival fraction: {100*doa_fraction:.0f}% (< 50%); width not the primary problem.")
        log.info(f"  Do NOT cherry-pick a single lucky cell. Experiment is complete.")

    # -----------------------------------------------------------------------
    # 10. Per-pair breakdown at SL=8, R=2 (current bracket proxy)
    # -----------------------------------------------------------------------
    log.info("\n[APPENDIX] Per-pair PF at SL=8, R=2.0 (current bracket proxy)")
    for epic in EXECUTION_IDS:
        pair = PAIR_NAMES[epic]
        cell = pd.DataFrame(results[(8, 2.0)])
        pair_cell = cell[cell['epic'] == epic]
        if len(pair_cell) == 0:
            continue
        pf_pair = compute_pf(pair_cell['pips'].tolist())
        wr_pair = sum(1 for p in pair_cell['pips'] if p > 0) / len(pair_cell)
        log.info(f"  {pair:8s}: n={len(pair_cell):3d}, PF={pf_pair:.3f}, WR={100*wr_pair:.1f}%")

    # -----------------------------------------------------------------------
    # 11. Timeout rates
    # -----------------------------------------------------------------------
    log.info("\n[APPENDIX] Timeout rates (72h horizon expired without resolution)")
    for sl in SL_VALUES:
        for r in R_MULTIPLES:
            t = n_timeout[(sl, r)]
            log.info(f"  SL={sl:>2} R={r:.1f}: {t:3d} timeouts ({100*t/1946:.1f}%)")

    log.info("\n=== SWEEP COMPLETE. NO DB ROWS WERE MODIFIED. ===")
    log.info("Script: /app/range_fade_width_sweep.py")


if __name__ == '__main__':
    run_sweep()
