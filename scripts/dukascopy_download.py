#!/usr/bin/env python3
"""
Dukascopy 1m forex/metal backfill downloader.

Pulls 1-minute OHLC candles from Dukascopy Bank's public data feed and
writes them in ig_candles_backtest CSV format (one file per epic).

Companion push script:  scripts/dukascopy_push_local.sh

Why this script exists
----------------------
The backtester reads from `ig_candles_backtest`, a derived/resampled
table that's excluded from Azure backups to save cost. This script is
the authoritative re-population mechanism: disaster-recovery procedure
is "restore DB from backup → re-run this + push → done."

Key design points
-----------------
- **UTC throughout**. Dukascopy is UTC — no EST/DST handling (unlike
  HistData).
- **Resume-friendly**. Per-epic CSV files with "complete" marker; rerun
  skips already-downloaded epics. Delete a CSV to re-fetch.
- **Week-chunked**. Each fetch call covers 1 week (~10k bars) to stay
  well under the library's 30k-row default cap and give progress output.
- **Gold is tagged with a distinct epic**. Dukascopy XAUUSD ≠ IG
  CS.D.CFEGOLD.CEE.IP pricing, so it lands as
  CS.D.CFEGOLD.DUKAS.IP to avoid polluting IG's live gold series.

Usage
-----
    # One-time host-side venv setup (scripts/ is not mounted in task-worker):
    python3 -m venv ~/.venvs/dukas
    ~/.venvs/dukas/bin/pip install dukascopy-python pandas

    # Backfill all configured epics, 2020-01-01 → end date
    ~/.venvs/dukas/bin/python scripts/dukascopy_download.py \\
        --start 2020-01-01 --end 2025-09-17 --output-dir /tmp/dukas/

    # Single epic
    ~/.venvs/dukas/bin/python scripts/dukascopy_download.py \\
        --epics CS.D.EURJPY.MINI.IP --start 2020-01-01 --end 2025-09-17 \\
        --output-dir /tmp/dukas/

    # Alternative — run inside task-worker (which already has deps installed):
    docker cp scripts/dukascopy_download.py task-worker:/tmp/
    docker exec task-worker python /tmp/dukascopy_download.py \\
        --start 2020-01-01 --end 2025-09-17 --output-dir /tmp/dukas/
    # Then copy CSVs out for the push step:
    docker cp task-worker:/tmp/dukas/ /tmp/

Next step: push the CSVs into ig_candles_backtest:
    ./scripts/dukascopy_push_local.sh /tmp/dukas/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd

try:
    import dukascopy_python
    from dukascopy_python import fetch, INTERVAL_MIN_1, OFFER_SIDE_BID
    from dukascopy_python.instruments import (
        INSTRUMENT_FX_MAJORS_EUR_USD,
        INSTRUMENT_FX_MAJORS_GBP_USD,
        INSTRUMENT_FX_MAJORS_USD_JPY,
        INSTRUMENT_FX_MAJORS_AUD_USD,
        INSTRUMENT_FX_MAJORS_USD_CHF,
        INSTRUMENT_FX_MAJORS_USD_CAD,
        INSTRUMENT_FX_MAJORS_NZD_USD,
        INSTRUMENT_FX_CROSSES_EUR_JPY,
        INSTRUMENT_FX_CROSSES_AUD_JPY,
        INSTRUMENT_FX_CROSSES_GBP_JPY,
        INSTRUMENT_FX_CROSSES_EUR_GBP,
        INSTRUMENT_FX_METALS_XAU_USD,
    )
except ImportError as e:
    sys.stderr.write(
        f"ERROR: dukascopy-python not installed ({e})\n"
        "Install with: docker exec task-worker pip install dukascopy-python\n"
    )
    sys.exit(1)


# ============================================================================
# Epic ↔ Dukascopy instrument mapping
# ============================================================================

@dataclass(frozen=True)
class EpicMapping:
    epic: str
    instrument: str
    description: str


EPIC_MAPPINGS: List[EpicMapping] = [
    # Majors
    EpicMapping("CS.D.EURUSD.CEEM.IP", INSTRUMENT_FX_MAJORS_EUR_USD, "EURUSD"),
    EpicMapping("CS.D.GBPUSD.MINI.IP", INSTRUMENT_FX_MAJORS_GBP_USD, "GBPUSD"),
    EpicMapping("CS.D.USDJPY.MINI.IP", INSTRUMENT_FX_MAJORS_USD_JPY, "USDJPY"),
    EpicMapping("CS.D.AUDUSD.MINI.IP", INSTRUMENT_FX_MAJORS_AUD_USD, "AUDUSD"),
    EpicMapping("CS.D.USDCHF.MINI.IP", INSTRUMENT_FX_MAJORS_USD_CHF, "USDCHF"),
    EpicMapping("CS.D.USDCAD.MINI.IP", INSTRUMENT_FX_MAJORS_USD_CAD, "USDCAD"),
    EpicMapping("CS.D.NZDUSD.MINI.IP", INSTRUMENT_FX_MAJORS_NZD_USD, "NZDUSD"),
    # JPY crosses
    EpicMapping("CS.D.EURJPY.MINI.IP", INSTRUMENT_FX_CROSSES_EUR_JPY, "EURJPY"),
    EpicMapping("CS.D.AUDJPY.MINI.IP", INSTRUMENT_FX_CROSSES_AUD_JPY, "AUDJPY"),
    EpicMapping("CS.D.GBPJPY.MINI.IP", INSTRUMENT_FX_CROSSES_GBP_JPY, "GBPJPY"),
    # Other crosses
    EpicMapping("CS.D.EURGBP.MINI.IP", INSTRUMENT_FX_CROSSES_EUR_GBP, "EURGBP"),
    # Metals — distinct epic to avoid mixing Dukascopy XAUUSD with IG CFEGOLD pricing
    EpicMapping("CS.D.CFEGOLD.DUKAS.IP", INSTRUMENT_FX_METALS_XAU_USD, "XAUUSD (gold)"),
]

EPIC_INDEX = {m.epic: m for m in EPIC_MAPPINGS}


# ============================================================================
# Fetch + CSV output
# ============================================================================

FETCH_WEEK_CHUNK = timedelta(days=7)
FETCH_MAX_RETRIES = 7
SLEEP_BETWEEN_CHUNKS_S = 0.3  # courteous pacing

CSV_HEADER = [
    "start_time", "epic", "timeframe",
    "open", "high", "low", "close",
    "volume", "ltv", "resampled_from",
]


def fetch_chunk(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch one chunk from Dukascopy. Returns empty DataFrame on any failure."""
    try:
        df = fetch(
            instrument,
            INTERVAL_MIN_1,
            OFFER_SIDE_BID,
            start,
            end,
            max_retries=FETCH_MAX_RETRIES,
        )
        return df
    except Exception as e:
        sys.stderr.write(f"    WARN: fetch failed {start} → {end}: {e}\n")
        return pd.DataFrame()


def download_epic(
    mapping: EpicMapping,
    start: datetime,
    end: datetime,
    output_path: str,
    *,
    overwrite: bool = False,
) -> int:
    """Download 1m bars for one epic and write CSV. Returns row count."""
    if os.path.exists(output_path) and not overwrite:
        # Quick sanity check: file has > 2 lines
        with open(output_path) as f:
            head = [next(f, None) for _ in range(3)]
        if head[1] is not None and head[2] is not None:
            # File looks populated; skip
            size = os.path.getsize(output_path)
            print(f"  SKIP {mapping.epic} ({mapping.description}): existing CSV "
                  f"{size // (1024*1024)} MB — delete to re-fetch")
            return 0

    print(f"  FETCH {mapping.epic} ({mapping.description}): {start.date()} → {end.date()}")
    t0 = time.time()

    frames: List[pd.DataFrame] = []
    cur = start
    chunks_done = 0
    total_weeks = max(1, int((end - start).total_seconds() // FETCH_WEEK_CHUNK.total_seconds()))

    while cur < end:
        chunk_end = min(cur + FETCH_WEEK_CHUNK, end)
        df = fetch_chunk(mapping.instrument, cur, chunk_end)
        if not df.empty:
            frames.append(df)
        chunks_done += 1
        if chunks_done % 13 == 0:
            pct = 100 * chunks_done / total_weeks
            print(f"    {cur.date()}  {chunks_done}/{total_weeks} weeks  {pct:.0f}%")
        cur = chunk_end
        time.sleep(SLEEP_BETWEEN_CHUNKS_S)

    if not frames:
        print(f"    EMPTY — no data returned for {mapping.epic}")
        return 0

    df_all = pd.concat(frames)
    # Drop duplicates on index (chunk boundaries can double-count last bar)
    df_all = df_all[~df_all.index.duplicated(keep="first")].sort_index()

    # Transform to ig_candles_backtest schema
    rows = len(df_all)
    out = pd.DataFrame({
        "start_time": df_all.index.tz_convert("UTC").tz_localize(None),
        "epic": mapping.epic,
        "timeframe": 1,
        "open": df_all["open"],
        "high": df_all["high"],
        "low": df_all["low"],
        "close": df_all["close"],
        "volume": df_all["volume"].fillna(0).round().astype("int64"),
        "ltv": 0,
        "resampled_from": 1,
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False, columns=CSV_HEADER, quoting=csv.QUOTE_MINIMAL)

    elapsed = time.time() - t0
    print(f"    wrote {rows:,} rows -> {output_path} ({elapsed:.0f}s)")
    return rows


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, UTC)")
    p.add_argument("--end", required=True, help="End date, exclusive (YYYY-MM-DD, UTC)")
    p.add_argument("--output-dir", default="/tmp/dukas/", help="Directory for per-epic CSVs")
    p.add_argument("--epics", nargs="*", default=None,
                   help="Specific epics to download (default: all configured)")
    p.add_argument("--list-epics", action="store_true",
                   help="Print the configured epic→instrument mapping and exit")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if CSV already exists")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_epics:
        print("Configured epic ↔ Dukascopy instrument mapping:")
        for m in EPIC_MAPPINGS:
            print(f"  {m.epic:28s}  {m.description:16s}  {m.instrument}")
        return 0

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.epics:
        missing = [e for e in args.epics if e not in EPIC_INDEX]
        if missing:
            sys.stderr.write(f"ERROR: unknown epic(s): {missing}\n")
            sys.stderr.write(f"       available: {list(EPIC_INDEX.keys())}\n")
            return 1
        targets = [EPIC_INDEX[e] for e in args.epics]
    else:
        targets = list(EPIC_MAPPINGS)

    print(f"Dukascopy backfill: {start.date()} → {end.date()} UTC")
    print(f"Output dir: {args.output_dir}")
    print(f"Epics: {len(targets)}  ({'overwrite' if args.overwrite else 'resume'})")
    print("=" * 72)

    total_rows = 0
    t0 = time.time()
    for m in targets:
        safe_name = m.epic.replace(".", "_")
        out_path = os.path.join(args.output_dir, f"{safe_name}.csv")
        try:
            total_rows += download_epic(m, start, end, out_path, overwrite=args.overwrite)
        except Exception as e:
            sys.stderr.write(f"ERROR on {m.epic}: {e}\n")
            continue

    elapsed = time.time() - t0
    print("=" * 72)
    print(f"Total new rows: {total_rows:,}  in {elapsed/60:.1f} min")
    print(f"Next step: scripts/dukascopy_push_local.sh {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
