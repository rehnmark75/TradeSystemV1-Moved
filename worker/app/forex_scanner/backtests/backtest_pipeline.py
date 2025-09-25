#!/usr/bin/env python3
"""
General Pipeline Backtest - Full Scanner Pipeline Testing
=========================================================

Run comprehensive backtests using the complete production scanner pipeline
with the ability to test individual strategies or all strategies together.

USAGE EXAMPLES:
  # Test individual strategies
  python backtest_pipeline.py --strategy-filter ema --start-date 2024-12-01 --end-date 2024-12-02
  python backtest_pipeline.py --strategy-filter momentum --start-date 2024-12-01 --end-date 2024-12-02
  python backtest_pipeline.py --strategy-filter smc --start-date 2024-12-01 --end-date 2024-12-02

  # Test all strategies
  python backtest_pipeline.py --strategy-filter all --start-date 2024-12-01 --end-date 2024-12-02

  # Test with specific epic
  python backtest_pipeline.py --strategy-filter ema --epic EURUSD.CEEM.IP --start-date 2024-12-01 --end-date 2024-12-02

FEATURES:
- Full production scanner pipeline (IntelligentForexScanner → SignalProcessor → EnhancedSignalValidator → TradeValidator → TradingOrchestrator)
- Historical data from ig_candles table (no mock data)
- Complete validation and decision-making process
- Strategy isolation testing (test one strategy at a time)
- Trade decision logging instead of execution
- Real market conditions simulation
- Comprehensive performance analysis
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

sys.path.insert(0, project_root)

try:
    from core.database import DatabaseManager
    from core.backtest.historical_scanner_engine import HistoricalScannerEngine
    from core.backtest.performance_analyzer import PerformanceAnalyzer
    from core.backtest.signal_analyzer import SignalAnalyzer
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.backtest.historical_scanner_engine import HistoricalScannerEngine
    from forex_scanner.core.backtest.performance_analyzer import PerformanceAnalyzer
    from forex_scanner.core.backtest.signal_analyzer import SignalAnalyzer
    from forex_scanner import config


class PipelineBacktest:
    """General Pipeline Backtesting using Historical Scanner Engine"""

    def __init__(self, epic_list: List[str] = None, user_timezone: str = 'Europe/Stockholm'):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.DATABASE_URL)

        # Epic list configuration
        if epic_list:
            self.epic_list = epic_list
        else:
            # Default epic list with proper naming
            self.epic_list = [
                'CS.D.EURUSD.CEEM.IP',  # Only EURUSD uses CEEM
                'CS.D.GBPUSD.MINI.IP',
                'CS.D.USDJPY.MINI.IP',
                'CS.D.USDCHF.MINI.IP',
                'CS.D.AUDUSD.MINI.IP',
                'CS.D.USDCAD.MINI.IP',
                'CS.D.NZDUSD.MINI.IP'
            ]

        self.user_timezone = user_timezone

        # Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()

    def run_pipeline_backtest(self,
                            start_date: datetime,
                            end_date: datetime,
                            timeframe: str = '15m',
                            strategy_filter: str = 'all') -> Dict:
        """
        Run backtest using the complete scanner pipeline

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Data timeframe (15m, 1h, etc.)
            strategy_filter: Strategy to test ('ema', 'momentum', 'smc', 'all', etc.)

        Returns:
            Comprehensive backtest results
        """

        self.logger.info(f"🚀 Starting Pipeline Backtest")
        self.logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        self.logger.info(f"⚡ Strategy Filter: {strategy_filter}")
        self.logger.info(f"🎯 Epics: {len(self.epic_list)}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")

        # Initialize Historical Scanner Engine
        historical_engine = HistoricalScannerEngine(
            db_manager=self.db_manager,
            epic_list=self.epic_list,
            user_timezone=self.user_timezone,
            strategy_filter=strategy_filter,  # Key parameter for strategy isolation
            scan_interval=900  # 15 minutes
        )

        # Run the complete historical backtest
        results = historical_engine.run_historical_backtest(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )

        if not results.get('success'):
            self.logger.error(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")
            return results

        # Generate additional analysis
        self._add_performance_analysis(results)

        return results

    def _add_performance_analysis(self, results: Dict):
        """Add comprehensive performance analysis to results"""

        trades = results.get('trades', [])
        if not trades:
            self.logger.warning("⚠️ No trades to analyze")
            return

        # Extract trade data for analysis
        trade_data = []
        for trade in trades:
            signal = trade.get('signal', {})
            trade_data.append({
                'timestamp': trade.get('historical_timestamp'),
                'epic': signal.get('epic'),
                'action': trade.get('action'),
                'strategy': signal.get('strategy'),
                'confidence': signal.get('confidence', 0),
                'price': signal.get('current_price', 0)
            })

        if trade_data:
            df_trades = pd.DataFrame(trade_data)

            # Add performance metrics
            results['performance_analysis'] = {
                'total_signals': len(df_trades),
                'approved_trades': len(df_trades[df_trades['action'].isin(['BUY', 'SELL'])]),
                'rejected_trades': len(df_trades[~df_trades['action'].isin(['BUY', 'SELL'])]),
                'by_strategy': df_trades['strategy'].value_counts().to_dict() if 'strategy' in df_trades.columns else {},
                'by_epic': df_trades['epic'].value_counts().to_dict() if 'epic' in df_trades.columns else {},
                'by_action': df_trades['action'].value_counts().to_dict() if 'action' in df_trades.columns else {},
                'avg_confidence': df_trades['confidence'].mean() if 'confidence' in df_trades.columns and len(df_trades) > 0 else 0
            }

            self.logger.info("✅ Performance analysis complete")
        else:
            self.logger.warning("⚠️ No valid trade data for performance analysis")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="General Pipeline Backtest - Test any strategy through complete scanner pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
STRATEGY FILTER OPTIONS:
  all              - Test all enabled strategies
  ema              - EMA Strategy only
  momentum         - Momentum Strategy only
  macd             - MACD Strategy only
  zero_lag         - Zero Lag Strategy only
  smc              - Smart Money Concepts Strategy only
  ichimoku         - Ichimoku Strategy only
  mean_reversion   - Mean Reversion Strategy only
  ranging_market   - Ranging Market Strategy only
  combined         - Combined Strategy only

EXAMPLES:
  python backtest_pipeline.py --strategy-filter ema --start-date 2024-12-01 --end-date 2024-12-02
  python backtest_pipeline.py --strategy-filter momentum --days 7
  python backtest_pipeline.py --strategy-filter all --epic EURUSD.CEEM.IP --days 30
        """
    )

    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days', type=int,
                          help='Number of days to backtest (from today backwards)')
    date_group.add_argument('--start-date', type=str,
                          help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD) - only used with --start-date')

    # Strategy and epic options
    parser.add_argument('--strategy-filter', type=str, default='all',
                       choices=['all', 'ema', 'momentum', 'macd', 'zero_lag', 'smc',
                               'ichimoku', 'mean_reversion', 'ranging_market', 'combined'],
                       help='Strategy to test (default: all)')

    parser.add_argument('--epic', type=str,
                       help='Single epic to test (e.g., EURUSD.CEEM.IP)')

    # Technical options
    parser.add_argument('--timeframe', type=str, default='15m',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Timeframe for analysis (default: 15m)')

    parser.add_argument('--timezone', type=str, default='Europe/Stockholm',
                       help='User timezone (default: Europe/Stockholm)')

    # Output options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main execution function"""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Calculate date range
    if args.days:
        # Auto-detect available data range and use the most recent data
        try:
            # Use the same import strategy as the rest of the file
            try:
                from core.database import DatabaseManager
            except ImportError:
                from forex_scanner.core.database import DatabaseManager

            db_manager = DatabaseManager(config.DATABASE_URL)

            # Query to find the latest available data range
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT MAX(start_time) as latest_time, MIN(start_time) as earliest_time
                        FROM ig_candles
                        WHERE timeframe = %s
                        AND epic IN ('CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP')
                    """, [{'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}.get(args.timeframe, 15)])
                    result = cursor.fetchone()

                    if result and result[0] and result[1]:
                        latest_time, earliest_time = result
                        # Use the available data range, but limit to requested days
                        end_date = latest_time
                        requested_start = latest_time - timedelta(days=args.days)
                        start_date = max(requested_start, earliest_time)  # Don't go earlier than available data

                        logger.info(f"📅 Auto-detected data range: {start_date} to {end_date}")
                        logger.info(f"🔍 Available data from {earliest_time} to {latest_time}")
                    else:
                        logger.error("❌ No data found in ig_candles table")
                        return 1
        except Exception as e:
            logger.error(f"❌ Error detecting data range: {e}")
            # Fallback to manual date calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            logger.warning(f"⚠️ Using fallback date range: {start_date} to {end_date}")
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()

    # Epic list configuration
    epic_list = None
    if args.epic:
        epic_list = [args.epic]
        logger.info(f"🎯 Testing single epic: {args.epic}")

    try:
        # Initialize and run backtest
        backtest = PipelineBacktest(
            epic_list=epic_list,
            user_timezone=args.timezone
        )

        results = backtest.run_pipeline_backtest(
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe,
            strategy_filter=args.strategy_filter
        )

        # Display results summary
        if results.get('success'):
            stats = results.get('statistics', {})
            perf = results.get('performance_analysis', {})

            logger.info("=" * 60)
            logger.info("📊 PIPELINE BACKTEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Strategy Filter: {args.strategy_filter}")
            logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"Total Scans: {stats.get('total_scans', 0):,}")
            logger.info(f"Signals Detected: {stats.get('signals_detected', 0):,}")
            logger.info(f"Trades Approved: {stats.get('trades_approved', 0):,}")
            logger.info(f"Trades Rejected: {stats.get('trades_rejected', 0):,}")
            logger.info(f"Approval Rate: {stats.get('approval_rate', 0):.1%}")
            logger.info(f"Duration: {stats.get('total_duration_seconds', 0):.1f}s")

            if perf:
                logger.info(f"Average Confidence: {perf.get('avg_confidence', 0):.2f}")

                if perf.get('by_strategy'):
                    logger.info("By Strategy:")
                    for strategy, count in perf['by_strategy'].items():
                        logger.info(f"  {strategy}: {count}")

                if perf.get('by_action'):
                    logger.info("By Action:")
                    for action, count in perf['by_action'].items():
                        logger.info(f"  {action}: {count}")

            logger.info("=" * 60)
        else:
            logger.error(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        logger.info("⚠️ Backtest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Backtest error: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())