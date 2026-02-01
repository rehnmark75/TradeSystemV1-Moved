#!/usr/bin/env python3
"""
Regime Period Analyzer

Identifies periods where specific market regimes dominated, useful for
testing parameter optimization strategies on regime-specific data.

This script queries the market_intelligence_history table to find date ranges
where a particular regime (trending, ranging, high_volatility, etc.) was
dominant for a specified percentage of the time.

Usage:
    # Find periods with at least 60% high_volatility regime in last 180 days
    python regime_period_analyzer.py --epic EURUSD --regime high_volatility --min-pct 60 --days 180

    # Find all regime-dominated periods for analysis
    python regime_period_analyzer.py --epic EURUSD --all-regimes --days 180

    # Output to JSON for pipeline processing
    python regime_period_analyzer.py --epic EURUSD --all-regimes --output-json

Available Regimes:
    - trending: Market showing clear directional movement
    - ranging: Market oscillating within bounds
    - breakout: Potential breakout conditions detected
    - reversal: Possible trend reversal forming
    - high_volatility: Above-average market volatility
    - low_volatility: Below-average market volatility
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection settings
DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'forex',
    'user': 'postgres',
    'password': 'postgres'
}

# All supported regimes
ALL_REGIMES = [
    'trending', 'ranging', 'breakout',
    'reversal', 'high_volatility', 'low_volatility'
]


@dataclass
class RegimePeriod:
    """Represents a period dominated by a specific regime"""
    regime: str
    start_date: str
    end_date: str
    days: int
    regime_pct: float
    total_records: int
    regime_records: int
    avg_confidence: float
    sessions_breakdown: Dict[str, int]


@dataclass
class RegimeAnalysis:
    """Complete regime analysis results"""
    epic: str
    analysis_days: int
    analysis_start: str
    analysis_end: str
    regime_distribution: Dict[str, Dict]
    dominated_periods: List[RegimePeriod]
    daily_regime_data: List[Dict]


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)


def get_epic_regime_distribution(
    epic: str,
    days: int = 180,
    conn=None
) -> Dict[str, Dict]:
    """
    Get regime distribution statistics for an epic over specified period.

    Args:
        epic: Currency pair (e.g., 'EURUSD' or 'CS.D.EURUSD.CEEM.IP')
        days: Number of days to analyze
        conn: Optional existing database connection

    Returns:
        Dict with regime stats: {regime: {count, pct, avg_confidence, ...}}
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        # Normalize epic format
        clean_epic = normalize_epic(epic)

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get overall regime distribution
        query = """
        WITH regime_data AS (
            SELECT
                dominant_regime,
                regime_confidence,
                current_session,
                scan_timestamp
            FROM market_intelligence_history
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
        )
        SELECT
            dominant_regime,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct,
            AVG(regime_confidence) as avg_confidence,
            MIN(regime_confidence) as min_confidence,
            MAX(regime_confidence) as max_confidence,
            MIN(scan_timestamp) as first_seen,
            MAX(scan_timestamp) as last_seen
        FROM regime_data
        GROUP BY dominant_regime
        ORDER BY count DESC
        """

        cursor.execute(query, (days,))
        rows = cursor.fetchall()

        distribution = {}
        for row in rows:
            distribution[row['dominant_regime']] = {
                'count': row['count'],
                'pct': float(row['pct']),
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.5,
                'min_confidence': float(row['min_confidence']) if row['min_confidence'] else 0.0,
                'max_confidence': float(row['max_confidence']) if row['max_confidence'] else 1.0,
                'first_seen': row['first_seen'].isoformat() if row['first_seen'] else None,
                'last_seen': row['last_seen'].isoformat() if row['last_seen'] else None
            }

        cursor.close()
        return distribution

    finally:
        if should_close:
            conn.close()


def find_regime_dominated_periods(
    epic: str,
    regime: str,
    days: int = 180,
    min_regime_pct: float = 60.0,
    min_period_days: int = 3,
    conn=None
) -> List[RegimePeriod]:
    """
    Find date ranges where a specific regime dominated for X% of the time.

    Uses a simple contiguous day approach: finds consecutive days where the
    target regime had >= min_regime_pct dominance each day.

    Args:
        epic: Currency pair to analyze
        regime: Target regime (e.g., 'high_volatility', 'trending')
        days: Historical days to search
        min_regime_pct: Minimum percentage of daily records for regime (default 60%)
        min_period_days: Minimum consecutive days for a valid period (default 3)
        conn: Optional database connection

    Returns:
        List of RegimePeriod objects representing dominated periods
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get daily regime breakdown - simpler query focusing on dominant regime per day
        query = """
        WITH daily_regime_counts AS (
            SELECT
                scan_timestamp::date as day,
                dominant_regime,
                COUNT(*) as count,
                AVG(regime_confidence) as avg_confidence
            FROM market_intelligence_history
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY scan_timestamp::date, dominant_regime
        ),
        daily_totals AS (
            SELECT
                day,
                SUM(count) as total_records
            FROM daily_regime_counts
            GROUP BY day
        ),
        daily_regime_pct AS (
            SELECT
                d.day,
                d.dominant_regime,
                d.count as regime_count,
                t.total_records,
                d.count * 100.0 / t.total_records as regime_pct,
                d.avg_confidence
            FROM daily_regime_counts d
            JOIN daily_totals t ON d.day = t.day
        )
        SELECT
            day,
            dominant_regime,
            regime_count,
            total_records,
            regime_pct,
            avg_confidence
        FROM daily_regime_pct
        WHERE dominant_regime = %s
        ORDER BY day ASC
        """

        cursor.execute(query, (days, regime))
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            return []

        # Build daily data for target regime only
        daily_data = {}
        for row in rows:
            daily_data[row['day']] = {
                'regime_count': row['regime_count'],
                'total_records': row['total_records'],
                'regime_pct': float(row['regime_pct']),
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.5
            }

        # Find contiguous periods where regime dominates each day
        periods = []
        sorted_days = sorted(daily_data.keys())

        i = 0
        while i < len(sorted_days):
            day = sorted_days[i]
            day_data = daily_data[day]

            # Check if this day meets the threshold
            if day_data['regime_pct'] >= min_regime_pct:
                # Start a new period
                period_start = i
                period_regime_records = day_data['regime_count']
                period_total_records = day_data['total_records']
                period_confidence_sum = day_data['avg_confidence'] * day_data['regime_count']

                # Extend period while consecutive days meet threshold
                j = i + 1
                while j < len(sorted_days):
                    next_day = sorted_days[j]
                    # Check for consecutive days (allow for weekends - up to 3 day gaps)
                    day_gap = (next_day - sorted_days[j-1]).days
                    if day_gap > 3:
                        break  # Too big a gap

                    next_data = daily_data.get(next_day)
                    if next_data and next_data['regime_pct'] >= min_regime_pct:
                        period_regime_records += next_data['regime_count']
                        period_total_records += next_data['total_records']
                        period_confidence_sum += next_data['avg_confidence'] * next_data['regime_count']
                        j += 1
                    else:
                        break  # Day doesn't meet threshold

                # Check if period meets minimum length
                period_days = j - period_start
                if period_days >= min_period_days:
                    final_pct = (period_regime_records / period_total_records * 100) if period_total_records > 0 else 0
                    avg_conf = period_confidence_sum / period_regime_records if period_regime_records > 0 else 0.5

                    period = RegimePeriod(
                        regime=regime,
                        start_date=sorted_days[period_start].isoformat(),
                        end_date=sorted_days[j - 1].isoformat(),
                        days=period_days,
                        regime_pct=round(final_pct, 1),
                        total_records=period_total_records,
                        regime_records=period_regime_records,
                        avg_confidence=round(avg_conf, 4),
                        sessions_breakdown={}
                    )
                    periods.append(period)

                i = j  # Move past this period
            else:
                i += 1  # Move to next day

        cursor.close()
        return periods

    finally:
        if should_close:
            conn.close()


def get_daily_regime_data(
    epic: str,
    days: int = 180,
    conn=None
) -> List[Dict]:
    """
    Get daily regime data for detailed analysis.

    Returns per-day breakdown of regime distribution for charting and analysis.
    """
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        WITH daily_data AS (
            SELECT
                scan_timestamp::date as day,
                dominant_regime,
                COUNT(*) as count,
                AVG(regime_confidence) as avg_confidence
            FROM market_intelligence_history
            WHERE scan_timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY scan_timestamp::date, dominant_regime
        ),
        daily_totals AS (
            SELECT
                day,
                SUM(count) as total_records
            FROM daily_data
            GROUP BY day
        )
        SELECT
            dd.day,
            dd.dominant_regime as regime,
            dd.count,
            dt.total_records,
            dd.count * 100.0 / dt.total_records as pct,
            dd.avg_confidence
        FROM daily_data dd
        JOIN daily_totals dt ON dd.day = dt.day
        ORDER BY dd.day ASC, dd.count DESC
        """

        cursor.execute(query, (days,))
        rows = cursor.fetchall()

        daily_data = []
        for row in rows:
            daily_data.append({
                'day': row['day'].isoformat(),
                'regime': row['regime'],
                'count': row['count'],
                'total_records': row['total_records'],
                'pct': round(float(row['pct']), 1),
                'avg_confidence': round(float(row['avg_confidence']), 4) if row['avg_confidence'] else 0.5
            })

        cursor.close()
        return daily_data

    finally:
        if should_close:
            conn.close()


def normalize_epic(epic: str) -> str:
    """Normalize epic to clean format (e.g., EURUSD)"""
    if epic.startswith('CS.D.'):
        # Extract pair name from full epic
        parts = epic.split('.')
        if len(parts) >= 3:
            return parts[2]
    return epic.upper()


def analyze_epic_regimes(
    epic: str,
    days: int = 180,
    regimes: Optional[List[str]] = None,
    min_regime_pct: float = 60.0,
    min_period_days: int = 3
) -> RegimeAnalysis:
    """
    Perform complete regime analysis for an epic.

    Args:
        epic: Currency pair to analyze
        days: Number of historical days
        regimes: List of regimes to find periods for (None = all)
        min_regime_pct: Minimum dominance percentage
        min_period_days: Minimum period length in days

    Returns:
        RegimeAnalysis with complete results
    """
    conn = get_db_connection()

    try:
        clean_epic = normalize_epic(epic)

        # Get overall distribution
        distribution = get_epic_regime_distribution(clean_epic, days, conn)

        # Find dominated periods for each regime
        target_regimes = regimes if regimes else ALL_REGIMES
        all_periods = []

        for regime in target_regimes:
            periods = find_regime_dominated_periods(
                clean_epic,
                regime,
                days,
                min_regime_pct,
                min_period_days,
                conn=conn
            )
            all_periods.extend(periods)

        # Sort periods by date
        all_periods.sort(key=lambda p: p.start_date)

        # Get daily data
        daily_data = get_daily_regime_data(clean_epic, days, conn)

        # Calculate analysis window
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        return RegimeAnalysis(
            epic=clean_epic,
            analysis_days=days,
            analysis_start=start_date.date().isoformat(),
            analysis_end=end_date.date().isoformat(),
            regime_distribution=distribution,
            dominated_periods=all_periods,
            daily_regime_data=daily_data
        )

    finally:
        conn.close()


def print_regime_analysis(analysis: RegimeAnalysis, verbose: bool = False):
    """Print regime analysis results in human-readable format"""

    print(f"\n{'='*70}")
    print(f"REGIME ANALYSIS: {analysis.epic}")
    print(f"{'='*70}")
    print(f"Period: {analysis.analysis_start} to {analysis.analysis_end} ({analysis.analysis_days} days)")

    # Distribution summary
    print(f"\n{'='*70}")
    print("REGIME DISTRIBUTION")
    print(f"{'='*70}")
    print(f"{'Regime':<18} {'Count':>10} {'Pct':>8} {'Avg Conf':>10}")
    print("-" * 50)

    total_count = sum(r['count'] for r in analysis.regime_distribution.values())
    for regime, stats in sorted(analysis.regime_distribution.items(), key=lambda x: -x[1]['count']):
        print(f"{regime:<18} {stats['count']:>10} {stats['pct']:>7.1f}% {stats['avg_confidence']:>9.2f}")

    print(f"\nTotal records: {total_count}")

    # Dominated periods
    if analysis.dominated_periods:
        print(f"\n{'='*70}")
        print("REGIME-DOMINATED PERIODS")
        print(f"{'='*70}")
        print(f"{'Regime':<18} {'Start':>12} {'End':>12} {'Days':>6} {'Pct':>7} {'Records':>8}")
        print("-" * 70)

        for period in analysis.dominated_periods:
            print(f"{period.regime:<18} {period.start_date:>12} {period.end_date:>12} "
                  f"{period.days:>6} {period.regime_pct:>6.1f}% {period.regime_records:>8}")

        print(f"\nTotal periods found: {len(analysis.dominated_periods)}")
    else:
        print("\nNo regime-dominated periods found with current thresholds.")

    # Verbose daily data
    if verbose and analysis.daily_regime_data:
        print(f"\n{'='*70}")
        print("DAILY REGIME BREAKDOWN (last 30 days)")
        print(f"{'='*70}")

        # Group by day and show top regime
        days_shown = {}
        for entry in analysis.daily_regime_data[-90:]:  # Last 30 days worth
            day = entry['day']
            if day not in days_shown:
                days_shown[day] = entry
            elif entry['pct'] > days_shown[day]['pct']:
                days_shown[day] = entry

        print(f"{'Day':>12} {'Top Regime':>18} {'Pct':>8} {'Confidence':>12}")
        print("-" * 55)

        for day, entry in sorted(days_shown.items())[-30:]:
            print(f"{day:>12} {entry['regime']:>18} {entry['pct']:>7.1f}% {entry['avg_confidence']:>11.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze market regime periods for parameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find high volatility periods
  python regime_period_analyzer.py --epic EURUSD --regime high_volatility --days 180

  # Find all regime-dominated periods
  python regime_period_analyzer.py --epic EURUSD --all-regimes --days 180

  # Lower threshold to find more periods
  python regime_period_analyzer.py --epic EURUSD --all-regimes --min-pct 50

  # Output JSON for pipeline
  python regime_period_analyzer.py --epic EURUSD --all-regimes --output-json
        """
    )

    parser.add_argument('--epic', type=str, default='EURUSD',
                        help='Currency pair to analyze (default: EURUSD)')
    parser.add_argument('--regime', type=str, choices=ALL_REGIMES,
                        help='Specific regime to find periods for')
    parser.add_argument('--all-regimes', action='store_true',
                        help='Find periods for all regimes')
    parser.add_argument('--days', type=int, default=180,
                        help='Number of days to analyze (default: 180)')
    parser.add_argument('--min-pct', type=float, default=60.0,
                        help='Minimum regime percentage for dominated period (default: 60)')
    parser.add_argument('--min-days', type=int, default=3,
                        help='Minimum days for a valid period (default: 3)')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed daily breakdown')

    args = parser.parse_args()

    # Determine which regimes to analyze
    if args.regime:
        regimes = [args.regime]
    elif args.all_regimes:
        regimes = ALL_REGIMES
    else:
        # Default: show distribution only
        regimes = None

    try:
        analysis = analyze_epic_regimes(
            epic=args.epic,
            days=args.days,
            regimes=regimes,
            min_regime_pct=args.min_pct,
            min_period_days=args.min_days
        )

        if args.output_json:
            # Use custom JSON encoder for Decimal types
            from decimal import Decimal

            class DecimalEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Decimal):
                        return float(obj)
                    return super().default(obj)

            output = {
                'epic': analysis.epic,
                'analysis_days': analysis.analysis_days,
                'analysis_start': analysis.analysis_start,
                'analysis_end': analysis.analysis_end,
                'regime_distribution': analysis.regime_distribution,
                'dominated_periods': [asdict(p) for p in analysis.dominated_periods],
                'daily_regime_data': analysis.daily_regime_data if args.verbose else []
            }
            print(json.dumps(output, indent=2, cls=DecimalEncoder))
        else:
            print_regime_analysis(analysis, verbose=args.verbose)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
