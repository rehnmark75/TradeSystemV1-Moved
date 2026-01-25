#!/usr/bin/env python3
"""
Batch Claude Analysis Runner

Runs Claude AI analysis on scanner signals that haven't been analyzed yet.
Fetches all necessary data (technical + fundamental) and sends to Claude API.

Usage:
    docker exec task-worker python -m stock_scanner.scripts.run_batch_claude_analysis
    docker exec task-worker python -m stock_scanner.scripts.run_batch_claude_analysis --limit 10
    docker exec task-worker python -m stock_scanner.scripts.run_batch_claude_analysis --tier A --tier B
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.services.claude_stock_analyzer import StockClaudeAnalyzer
from stock_scanner.services.stock_response_parser import ClaudeAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("batch_claude_analysis")


class BatchClaudeAnalyzer:
    """
    Batch analyzer that fetches scanner signals and runs Claude analysis.

    Features:
    - Fetches unanalyzed signals from database
    - Gathers technical data from screening metrics
    - Gathers fundamental data from instruments table
    - Runs Claude analysis with comprehensive data
    - Saves results back to database
    """

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.claude = StockClaudeAnalyzer(db_manager=db_manager, enable_charts=True)

    async def get_pending_signals(
        self,
        tiers: Optional[List[str]] = None,
        limit: int = 50,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get scanner signals that haven't been analyzed by Claude yet.

        Args:
            tiers: Filter by quality tiers (e.g., ['A', 'B'])
            limit: Maximum signals to return
            days_back: How far back to look for signals

        Returns:
            List of signal dictionaries
        """
        tier_filter = ""
        if tiers:
            tier_list = ", ".join([f"'{t}'" for t in tiers])
            tier_filter = f"AND quality_tier IN ({tier_list})"

        query = f"""
            SELECT
                s.*
            FROM stock_scanner_signals s
            WHERE s.claude_analyzed_at IS NULL
              AND s.created_at >= NOW() - INTERVAL '{days_back} days'
              {tier_filter}
            ORDER BY s.composite_score DESC, s.created_at DESC
            LIMIT {limit}
        """

        rows = await self.db.fetch(query)
        return [dict(row) for row in rows]

    async def get_technical_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get technical data for a ticker from screening metrics.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with technical indicators
        """
        query = """
            SELECT
                m.current_price,
                m.atr_14,
                m.atr_percent,
                m.rsi_14,
                m.rsi_signal,
                m.macd,
                m.macd_signal,
                m.macd_histogram,
                m.macd_cross_signal,
                m.sma_20,
                m.sma_50,
                m.sma_200,
                m.sma_cross_signal,
                m.price_vs_sma20,
                m.price_vs_sma50,
                m.price_vs_sma200,
                m.avg_volume_20,
                m.relative_volume,
                m.percentile_volume,
                m.historical_volatility_20,
                m.high_52w,
                m.low_52w,
                m.pct_from_52w_high,
                m.pct_from_52w_low,
                m.high_low_signal,
                m.gap_percent,
                m.gap_signal,
                m.candlestick_pattern,
                m.price_change_1d,
                m.perf_1w,
                m.perf_1m,
                m.perf_ytd,
                m.trend_strength,
                m.ma_alignment,
                m.smc_trend,
                m.smc_bias,
                m.last_bos_type,
                m.last_bos_date,
                m.last_bos_price,
                m.swing_high,
                m.swing_low,
                m.premium_discount_zone,
                m.zone_position,
                m.nearest_ob_type,
                m.nearest_ob_price,
                m.nearest_ob_distance,
                m.smc_confluence_score,
                m.calculation_date
            FROM stock_screening_metrics m
            WHERE m.ticker = $1
            ORDER BY m.calculation_date DESC
            LIMIT 1
        """

        row = await self.db.fetchrow(query, ticker)

        if not row:
            logger.warning(f"No technical data found for {ticker}")
            return {}

        # Convert all values to native Python types (handle Decimal)
        data = {}
        for key, value in dict(row).items():
            if hasattr(value, '__float__'):
                data[key] = float(value)
            else:
                data[key] = value

        # Calculate derived fields for compatibility
        perf_1w = data.get('perf_1w') or 0
        data['price_change_5d'] = float(perf_1w) * 0.7  # Approx

        # Rename for compatibility with prompt builder
        data['close'] = data.get('current_price')
        data['perf_1d'] = data.get('price_change_1d')

        # Build SMC sub-dict for prompt builder
        if data.get('smc_trend'):
            data['smc'] = {
                'smc_trend': data.get('smc_trend'),
                'smc_bias': data.get('smc_bias'),
                'last_bos_type': data.get('last_bos_type'),
                'last_bos_date': data.get('last_bos_date'),
                'last_bos_price': data.get('last_bos_price'),
                'swing_high': data.get('swing_high'),
                'swing_low': data.get('swing_low'),
                'premium_discount_zone': data.get('premium_discount_zone'),
                'zone_position': data.get('zone_position'),
                'nearest_ob_type': data.get('nearest_ob_type'),
                'nearest_ob_price': data.get('nearest_ob_price'),
                'nearest_ob_distance': data.get('nearest_ob_distance'),
                'smc_confluence_score': data.get('smc_confluence_score'),
            }

        return data

    async def get_fundamental_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data for a ticker from instruments table.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with fundamental metrics
        """
        query = """
            SELECT
                i.name,
                i.sector,
                i.industry,
                i.exchange,
                i.market_cap,
                i.avg_volume,
                i.earnings_date,
                i.earnings_date_estimated,
                i.beta,
                i.short_ratio,
                i.short_percent_float,
                i.institutional_percent,
                i.insider_percent,
                i.forward_pe,
                i.trailing_pe,
                i.price_to_book,
                i.price_to_sales,
                i.peg_ratio,
                i.dividend_yield,
                i.dividend_rate,
                i.payout_ratio,
                i.analyst_rating,
                i.target_price,
                i.target_high,
                i.target_low,
                i.number_of_analysts,
                i.revenue_growth,
                i.earnings_growth,
                i.earnings_quarterly_growth,
                i.profit_margin,
                i.operating_margin,
                i.gross_margin,
                i.return_on_equity,
                i.return_on_assets,
                i.debt_to_equity,
                i.current_ratio,
                i.quick_ratio,
                i.enterprise_value,
                i.enterprise_to_ebitda,
                i.enterprise_to_revenue,
                i.fifty_two_week_high,
                i.fifty_two_week_low,
                i.fifty_two_week_change,
                i.shares_outstanding,
                i.shares_float,
                i.shares_short,
                i.employee_count,
                i.country,
                i.fundamentals_updated_at,
                i.business_summary
            FROM stock_instruments i
            WHERE i.ticker = $1
        """

        row = await self.db.fetchrow(query, ticker)

        if not row:
            logger.warning(f"No fundamental data found for {ticker}")
            return {}

        # Convert all values to native Python types (handle Decimal)
        data = {}
        for key, value in dict(row).items():
            if hasattr(value, '__float__'):
                data[key] = float(value)
            else:
                data[key] = value

        # Calculate days to earnings
        if data.get('earnings_date'):
            earnings_dt = data['earnings_date']
            if hasattr(earnings_dt, 'date'):
                days_to = (earnings_dt.date() - datetime.now().date()).days
                data['days_to_earnings'] = max(0, days_to)
            else:
                data['days_to_earnings'] = None
        else:
            data['days_to_earnings'] = None

        # Format market cap
        if data.get('market_cap'):
            mc = data['market_cap']
            if mc >= 1e12:
                data['market_cap_formatted'] = f"${mc/1e12:.1f}T"
            elif mc >= 1e9:
                data['market_cap_formatted'] = f"${mc/1e9:.1f}B"
            elif mc >= 1e6:
                data['market_cap_formatted'] = f"${mc/1e6:.0f}M"
            else:
                data['market_cap_formatted'] = f"${mc:,.0f}"

        return data

    async def analyze_signal(
        self,
        signal: Dict[str, Any],
        analysis_level: str = 'comprehensive'
    ) -> Optional[ClaudeAnalysis]:
        """
        Analyze a single signal with all available data.

        Args:
            signal: Signal data from database
            analysis_level: 'quick', 'standard', or 'comprehensive'

        Returns:
            ClaudeAnalysis result or None on error
        """
        ticker = signal.get('ticker')

        # Fetch all data
        technical = await self.get_technical_data(ticker)
        fundamental = await self.get_fundamental_data(ticker)

        logger.info(
            f"Analyzing {ticker}: "
            f"score={signal.get('composite_score')}, "
            f"tier={signal.get('quality_tier')}, "
            f"technical_fields={len(technical)}, "
            f"fundamental_fields={len(fundamental)}"
        )

        # Run Claude analysis
        analysis = await self.claude.analyze_signal(
            signal=signal,
            technical_data=technical,
            fundamental_data=fundamental,
            analysis_level=analysis_level
        )

        return analysis

    async def save_analysis(
        self,
        signal_id: int,
        analysis: ClaudeAnalysis
    ) -> bool:
        """
        Save Claude analysis results to database.

        Args:
            signal_id: Signal ID to update
            analysis: ClaudeAnalysis result

        Returns:
            True if successful
        """
        query = """
            UPDATE stock_scanner_signals SET
                claude_grade = $2,
                claude_score = $3,
                claude_conviction = $4,
                claude_action = $5,
                claude_thesis = $6,
                claude_key_strengths = $7,
                claude_key_risks = $8,
                claude_position_rec = $9,
                claude_stop_adjustment = $10,
                claude_time_horizon = $11,
                claude_raw_response = $12,
                claude_analyzed_at = NOW(),
                claude_tokens_used = $13,
                claude_latency_ms = $14,
                claude_model = $15,
                updated_at = NOW()
            WHERE id = $1
        """

        try:
            await self.db.execute(
                query,
                signal_id,
                analysis.grade,
                analysis.score,
                analysis.conviction,
                analysis.action,
                analysis.thesis,
                analysis.key_strengths,
                analysis.key_risks,
                analysis.position_recommendation,
                analysis.stop_adjustment,
                analysis.time_horizon,
                analysis.raw_response,
                analysis.tokens_used,
                analysis.latency_ms,
                analysis.model
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis for signal {signal_id}: {e}")
            return False

    async def run_batch_analysis(
        self,
        tiers: Optional[List[str]] = None,
        limit: int = 50,
        analysis_level: str = 'comprehensive',
        delay_between: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run batch analysis on pending signals.

        Args:
            tiers: Filter by quality tiers
            limit: Maximum signals to analyze
            analysis_level: Analysis depth
            delay_between: Seconds between API calls

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("BATCH CLAUDE ANALYSIS")
        logger.info("=" * 60)

        # Check API availability
        if not self.claude.is_available:
            logger.error("Claude API not available. Check CLAUDE_API_KEY environment variable.")
            return {'error': 'API not available'}

        # Test connection
        logger.info("Testing Claude API connection...")
        if not self.claude.test_connection():
            logger.error("Claude API connection test failed")
            return {'error': 'Connection test failed'}
        logger.info("Claude API connection OK")

        # Fetch pending signals
        signals = await self.get_pending_signals(tiers=tiers, limit=limit)

        if not signals:
            logger.info("No pending signals to analyze")
            return {'analyzed': 0, 'message': 'No pending signals'}

        logger.info(f"Found {len(signals)} signals to analyze")

        # Process signals
        stats = {
            'total': len(signals),
            'analyzed': 0,
            'failed': 0,
            'skipped': 0,
            'by_grade': {},
            'by_action': {},
            'total_tokens': 0,
            'total_latency_ms': 0,
        }

        for i, signal in enumerate(signals, 1):
            ticker = signal.get('ticker')
            signal_id = signal.get('id')

            logger.info(f"\n[{i}/{len(signals)}] Analyzing {ticker}...")

            try:
                analysis = await self.analyze_signal(signal, analysis_level)

                if analysis:
                    # Save to database
                    saved = await self.save_analysis(signal_id, analysis)

                    if saved:
                        stats['analyzed'] += 1
                        stats['total_tokens'] += analysis.tokens_used or 0
                        stats['total_latency_ms'] += analysis.latency_ms or 0

                        # Track by grade
                        grade = analysis.grade
                        stats['by_grade'][grade] = stats['by_grade'].get(grade, 0) + 1

                        # Track by action
                        action = analysis.action
                        stats['by_action'][action] = stats['by_action'].get(action, 0) + 1

                        logger.info(
                            f"  ✓ {ticker}: Grade={analysis.grade}, "
                            f"Action={analysis.action}, "
                            f"Score={analysis.score}/10"
                        )
                    else:
                        stats['failed'] += 1
                        logger.error(f"  ✗ Failed to save analysis for {ticker}")
                else:
                    stats['failed'] += 1
                    logger.error(f"  ✗ No analysis returned for {ticker}")

            except Exception as e:
                stats['failed'] += 1
                logger.error(f"  ✗ Error analyzing {ticker}: {e}")

            # Delay between requests
            if i < len(signals):
                await asyncio.sleep(delay_between)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("BATCH ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Signals: {stats['total']}")
        logger.info(f"Analyzed: {stats['analyzed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total Tokens: {stats['total_tokens']:,}")
        logger.info(f"Avg Latency: {stats['total_latency_ms'] / max(1, stats['analyzed']):.0f}ms")

        if stats['by_grade']:
            logger.info("\nBy Grade:")
            for grade, count in sorted(stats['by_grade'].items()):
                logger.info(f"  {grade}: {count}")

        if stats['by_action']:
            logger.info("\nBy Action:")
            for action, count in sorted(stats['by_action'].items()):
                logger.info(f"  {action}: {count}")

        return stats


async def main():
    parser = argparse.ArgumentParser(description='Run batch Claude analysis on scanner signals')
    parser.add_argument('--limit', type=int, default=50, help='Max signals to analyze')
    parser.add_argument('--tier', action='append', dest='tiers', help='Filter by tier (can be repeated)')
    parser.add_argument('--level', choices=['quick', 'standard', 'comprehensive'],
                       default='comprehensive', help='Analysis depth')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls (seconds)')

    args = parser.parse_args()

    # Initialize database
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    try:
        analyzer = BatchClaudeAnalyzer(db)
        stats = await analyzer.run_batch_analysis(
            tiers=args.tiers,
            limit=args.limit,
            analysis_level=args.level,
            delay_between=args.delay
        )

        print(f"\nResults: {stats}")

    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
