#!/usr/bin/env python3
"""
HistData.com Forex Data Downloader

Downloads 1-minute forex data from HistData.com and transforms it
to ig_candles format for import into Azure PostgreSQL.

Usage:
    # Download 2025 data (filtered to Jan-Sept 17)
    docker exec task-worker python /app/forex_scanner/../scripts/histdata_download.py \
        --year 2025 --cutoff-date 2025-09-18 --output /tmp/histdata_2025.csv
"""

import argparse
import os
import sys
import tempfile
import zipfile
from datetime import datetime

import pandas as pd
from zoneinfo import ZoneInfo

# Pair mapping: HistData symbol -> IG Epic
PAIR_MAPPING = {
    'eurusd': 'CS.D.EURUSD.CEEM.IP',
    'gbpusd': 'CS.D.GBPUSD.MINI.IP',
    'usdjpy': 'CS.D.USDJPY.MINI.IP',
    'audusd': 'CS.D.AUDUSD.MINI.IP',
    'usdchf': 'CS.D.USDCHF.MINI.IP',
    'usdcad': 'CS.D.USDCAD.MINI.IP',
    'nzdusd': 'CS.D.NZDUSD.MINI.IP',
    'eurjpy': 'CS.D.EURJPY.MINI.IP',
    'audjpy': 'CS.D.AUDJPY.MINI.IP',
}

# All pairs we want to download
DEFAULT_PAIRS = list(PAIR_MAPPING.keys())


def download_histdata_year(pair: str, year: int, output_dir: str) -> str | None:
    """
    Download 1-minute data for a full year from HistData.

    Returns the path to the extracted CSV file, or None if download failed.
    """
    try:
        from histdata import download_hist_data
        from histdata.api import Platform, TimeFrame

        print(f"  Downloading {pair.upper()} {year}...")

        # Download full year (month=None for past/current years)
        download_hist_data(
            year=str(year),
            month=None,
            pair=pair,
            platform=Platform.GENERIC_ASCII,
            time_frame=TimeFrame.ONE_MINUTE,
            output_directory=output_dir
        )

        # Extract zip file if present
        for f in os.listdir(output_dir):
            if f.endswith('.zip') and pair.upper() in f.upper():
                zip_path = os.path.join(output_dir, f)
                print(f"    Extracting {f}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(output_dir)
                os.remove(zip_path)

        # Find the CSV file
        for f in os.listdir(output_dir):
            if f.endswith('.csv') and pair.upper() in f.upper():
                return os.path.join(output_dir, f)

        return None

    except ImportError:
        print("ERROR: histdata package not installed. Install with: pip install histdata")
        return None
    except Exception as e:
        print(f"  WARNING: Failed to download {pair} {year}: {e}")
        return None


def parse_histdata_csv(filepath: str) -> pd.DataFrame:
    """
    Parse HistData CSV format.

    HistData format (no headers, semicolon-separated):
    20250101 170000;1.03503;1.03514;1.03503;1.03514;0

    Columns: DateTime, Open, High, Low, Close, Volume
    """
    try:
        # Try semicolon separator first (most common)
        df = pd.read_csv(
            filepath,
            sep=';',
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume']
        )
    except Exception:
        # Try comma separator
        df = pd.read_csv(
            filepath,
            sep=',',
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume']
        )

    return df


def convert_est_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EST/EDT timestamps to UTC.

    HistData uses New York time (EST/EDT).
    - EST (Nov-Mar): UTC-5
    - EDT (Mar-Nov): UTC-4
    """
    # Parse datetime string: "20250101 170000" -> datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')

    # Localize to US/Eastern (handles DST automatically)
    eastern = ZoneInfo('America/New_York')
    utc = ZoneInfo('UTC')

    df['datetime'] = df['datetime'].apply(
        lambda x: x.replace(tzinfo=eastern).astimezone(utc).replace(tzinfo=None)
    )

    return df


def transform_to_ig_format(df: pd.DataFrame, epic: str) -> pd.DataFrame:
    """
    Transform HistData DataFrame to ig_candles format.

    Output columns: start_time, epic, timeframe, open, high, low, close, volume, ltv, data_source
    """
    result = pd.DataFrame({
        'start_time': df['datetime'],
        'epic': epic,
        'timeframe': 1,  # 1-minute
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume'].fillna(0).astype(int),
        'ltv': 0,  # Not available from HistData
        'data_source': 'histdata_backfill'
    })

    return result


def download_all_pairs(
    pairs: list[str],
    year: int,
    output_file: str,
    cutoff_date: datetime | None = None
) -> int:
    """
    Download data for all pairs for a full year, combine into single CSV.

    Returns total number of candles downloaded.
    """
    all_data = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for pair in pairs:
            epic = PAIR_MAPPING.get(pair.lower())
            if not epic:
                print(f"WARNING: Unknown pair {pair}, skipping")
                continue

            print(f"\nProcessing {pair.upper()} -> {epic}")

            # Create pair-specific temp dir
            pair_dir = os.path.join(temp_dir, f"{pair}_{year}")
            os.makedirs(pair_dir, exist_ok=True)

            csv_path = download_histdata_year(pair, year, pair_dir)

            if csv_path and os.path.exists(csv_path):
                try:
                    df = parse_histdata_csv(csv_path)
                    print(f"    Raw rows: {len(df):,}")
                    df = convert_est_to_utc(df)
                    df = transform_to_ig_format(df, epic)
                    all_data.append(df)
                    print(f"    Converted: {len(df):,} candles")
                except Exception as e:
                    print(f"    ERROR parsing: {e}")
            else:
                print(f"    No data available")

    if not all_data:
        print("\nERROR: No data downloaded!")
        return 0

    # Combine all pairs
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal before filtering: {len(final_df):,} candles")

    # Apply cutoff date filter if specified
    if cutoff_date:
        before_filter = len(final_df)
        final_df = final_df[final_df['start_time'] < cutoff_date]
        print(f"Filtered to before {cutoff_date}: {before_filter:,} -> {len(final_df):,} candles")

    # Sort by epic and time
    final_df = final_df.sort_values(['epic', 'start_time'])

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(final_df):,} candles to {output_file}")

    # Show summary
    print("\nSummary by pair:")
    summary = final_df.groupby('epic').agg({
        'start_time': ['min', 'max', 'count']
    })
    summary.columns = ['earliest', 'latest', 'count']
    print(summary.to_string())

    return len(final_df)


def main():
    parser = argparse.ArgumentParser(
        description='Download forex 1-minute data from HistData.com'
    )
    parser.add_argument(
        '--year', type=int, required=True,
        help='Year to download (e.g., 2025)'
    )
    parser.add_argument(
        '--pairs', type=str, nargs='+', default=DEFAULT_PAIRS,
        help='Pairs to download (default: all 9 forex pairs)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--cutoff-date', type=str, default=None,
        help='Only include data before this date (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    cutoff = None
    if args.cutoff_date:
        cutoff = datetime.strptime(args.cutoff_date, '%Y-%m-%d')

    print("HistData Forex Downloader")
    print("=" * 50)
    print(f"Year: {args.year}")
    print(f"Pairs: {[p.upper() for p in args.pairs]}")
    print(f"Output: {args.output}")
    if cutoff:
        print(f"Cutoff: {cutoff}")
    print()

    # Check if histdata package is installed
    try:
        import histdata
        print(f"Using histdata package version: {getattr(histdata, '__version__', 'unknown')}")
    except ImportError:
        print("ERROR: histdata package not installed!")
        print("Install with: pip install histdata")
        sys.exit(1)

    total = download_all_pairs(
        pairs=args.pairs,
        year=args.year,
        output_file=args.output,
        cutoff_date=cutoff
    )

    if total > 0:
        print(f"\n✅ Success! Downloaded {total:,} candles")
        print(f"\nNext step: Push to Azure with:")
        print(f"  ./scripts/azure_histdata_push.sh {args.output}")
    else:
        print("\n❌ Failed to download data")
        sys.exit(1)


if __name__ == '__main__':
    main()
