#!/usr/bin/env python3
"""
Run Claude AI analysis on stock signals with chart vision support.

Usage:
    docker exec task-worker python3 /app/stock_scanner/scripts/run_claude_reanalysis.py
    docker exec task-worker python3 /app/stock_scanner/scripts/run_claude_reanalysis.py --limit 10
    docker exec task-worker python3 /app/stock_scanner/scripts/run_claude_reanalysis.py --tier A+ --force
"""
import sys
sys.path.insert(0, '/app')

import asyncio
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from stock_scanner.scanners.scanner_manager import ScannerManager
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner import config


async def run_reanalysis(
    min_tier: str = 'A',
    max_signals: int = 10,
    force_reanalyze: bool = False,
    analysis_level: str = 'comprehensive'
):
    """
    Run Claude analysis with chart vision on stock signals.

    Args:
        min_tier: Minimum quality tier to analyze ('A+', 'A', 'B')
        max_signals: Maximum number of signals to analyze
        force_reanalyze: If True, reanalyze even already-analyzed signals
        analysis_level: 'quick', 'standard', or 'comprehensive'
    """
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    print("=" * 70)
    print(f"CLAUDE AI SIGNAL ANALYSIS WITH CHART VISION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Initialize scanner manager
    manager = ScannerManager(db_manager=db)

    # Get signals to analyze
    if force_reanalyze:
        # Get recent signals regardless of analysis status
        tier_filter = "('A+', 'A')" if min_tier == 'A' else f"('{min_tier}')"
        if min_tier == 'B':
            tier_filter = "('A+', 'A', 'B')"

        query = f"""
            SELECT * FROM stock_scanner_signals
            WHERE quality_tier IN {tier_filter}
            AND signal_timestamp > NOW() - INTERVAL '7 days'
            ORDER BY composite_score DESC
            LIMIT {max_signals}
        """
        rows = await db.fetch(query)
        signals = [dict(row) for row in rows]
        print(f"\nForce reanalyze mode: Got {len(signals)} recent {min_tier}+ signals")
    else:
        # Get unanalyzed signals only
        signals = await manager.get_unanalyzed_signals(min_tier=min_tier, limit=max_signals)
        print(f"\nFound {len(signals)} unanalyzed {min_tier}+ signals")

    if not signals:
        print("\n⚠️  No signals to analyze!")
        await db.close()
        return

    # Show signals to be analyzed
    print(f"\nSignals to analyze:")
    print("-" * 60)
    for s in signals:
        existing = s.get('claude_grade', '-')
        print(f"  {s['ticker']:6} | {s['signal_type']:4} | {s['quality_tier']:3} | "
              f"Score: {s['composite_score']:3} | Current: {existing}")
    print("-" * 60)

    # Run analysis with charts
    print(f"\n🔄 Running {analysis_level} analysis with chart vision...")
    print("   (This may take a few minutes)\n")

    results = await manager.analyze_signals_with_claude(
        signals=signals,
        min_tier=min_tier,
        max_signals=max_signals,
        analysis_level=analysis_level,
        model='sonnet'
    )

    # Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    upgrades = 0
    downgrades = 0

    for signal, analysis in results:
        ticker = signal.get('ticker', '???')
        scanner_score = signal.get('composite_score', 0)
        old_grade = signal.get('claude_grade')

        if analysis:
            new_grade = analysis.grade

            # Track grade changes
            grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, None: 0}
            if old_grade and grade_order.get(new_grade, 0) > grade_order.get(old_grade, 0):
                upgrades += 1
                change = "⬆️ UPGRADE"
            elif old_grade and grade_order.get(new_grade, 0) < grade_order.get(old_grade, 0):
                downgrades += 1
                change = "⬇️ DOWNGRADE"
            else:
                change = "➡️ NEW" if not old_grade else "↔️ SAME"

            print(f"\n{'='*60}")
            print(f"📊 {ticker} - {change}")
            print(f"{'='*60}")
            print(f"  Scanner Score: {scanner_score}/100 ({signal.get('quality_tier', 'N/A')} tier)")
            print(f"  Previous Grade: {old_grade or 'Not analyzed'}")
            print(f"  New Grade: {new_grade} (Claude Score: {analysis.score}/10)")
            print(f"  Action: {analysis.action}")
            print(f"  Conviction: {analysis.conviction}")
            print(f"  Position: {analysis.position_recommendation}")
            print(f"  Time Horizon: {analysis.time_horizon}")
            print(f"\n  📝 Thesis:")
            # Wrap thesis text
            thesis = analysis.thesis
            words = thesis.split()
            line = "     "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "     " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)

            if analysis.key_strengths:
                print(f"\n  ✅ Strengths:")
                for s in analysis.key_strengths[:3]:
                    print(f"     • {s}")

            if analysis.key_risks:
                print(f"\n  ⚠️  Risks:")
                for r in analysis.key_risks[:3]:
                    print(f"     • {r}")
        else:
            print(f"\n❌ {ticker}: Analysis failed")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total analyzed: {len(results)}")
    print(f"  Upgrades: {upgrades}")
    print(f"  Downgrades: {downgrades}")
    print(f"  Analysis level: {analysis_level}")
    print(f"  Charts included: Yes (vision enabled)")

    await db.close()
    print(f"\n✅ Analysis complete at {datetime.now().strftime('%H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description='Run Claude AI analysis on stock signals')
    parser.add_argument('--tier', '-t', default='A',
                       choices=['A+', 'A', 'B'],
                       help='Minimum quality tier to analyze (default: A)')
    parser.add_argument('--limit', '-l', type=int, default=10,
                       help='Maximum signals to analyze (default: 10)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force reanalyze already-analyzed signals')
    parser.add_argument('--level', default='comprehensive',
                       choices=['quick', 'standard', 'comprehensive'],
                       help='Analysis depth (default: comprehensive)')

    args = parser.parse_args()

    asyncio.run(run_reanalysis(
        min_tier=args.tier,
        max_signals=args.limit,
        force_reanalyze=args.force,
        analysis_level=args.level
    ))


if __name__ == '__main__':
    main()
