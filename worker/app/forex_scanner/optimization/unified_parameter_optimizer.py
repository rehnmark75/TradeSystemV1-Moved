#!/usr/bin/env python3
# optimization/unified_parameter_optimizer.py
"""
Unified Parameter Optimizer CLI

Consolidates data from all sources (trades, rejections, backtests, market intelligence)
to generate statistically validated, per-epic parameter recommendations with
direction-aware analysis and regime filter recommendations.

Usage:
    python unified_parameter_optimizer.py [options]

Examples:
    # Basic analysis (dry run)
    python unified_parameter_optimizer.py

    # Full analysis with direction and regime breakdown
    python unified_parameter_optimizer.py --days 30 --direction-analysis --include-regime-analysis -v

    # Export SQL to file
    python unified_parameter_optimizer.py --export-sql /app/optimization_updates.sql

    # Specific epic
    python unified_parameter_optimizer.py --epic EURUSD --direction-analysis
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.database import DatabaseManager
    import config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner import config

from optimization.data_collectors import TradeCollector, RejectionCollector, MarketIntelCollector
from optimization.analyzers import CorrelationAnalyzer, DirectionAnalyzer, RegimeAnalyzer
from optimization.generators import SQLGenerator, ReportGenerator


class UnifiedParameterOptimizer:
    """Main orchestrator for unified parameter optimization"""

    # Epic name to full epic mapping
    EPIC_MAPPING = {
        'EURUSD': 'CS.D.EURUSD.CEEM.IP',
        'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
        'USDJPY': 'CS.D.USDJPY.MINI.IP',
        'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
        'USDCHF': 'CS.D.USDCHF.MINI.IP',
        'USDCAD': 'CS.D.USDCAD.MINI.IP',
        'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
        'EURJPY': 'CS.D.EURJPY.CEEM.IP',
        'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
        'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
    }

    def __init__(
        self,
        days: int = 30,
        min_sample_size: int = 20,
        min_confidence: float = 0.70,
        epics: List[str] = None,
        include_direction_analysis: bool = False,
        include_regime_analysis: bool = False,
        verbose: bool = False
    ):
        self.days = days
        self.min_sample_size = min_sample_size
        self.min_confidence = min_confidence
        self.epics = epics
        self.include_direction_analysis = include_direction_analysis
        self.include_regime_analysis = include_regime_analysis
        self.verbose = verbose

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        # Forex database for trades, rejections, market intel
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.trade_collector = TradeCollector(self.db_manager)
        self.rejection_collector = RejectionCollector(self.db_manager)
        self.market_intel_collector = MarketIntelCollector(self.db_manager)

        # Strategy config database for pair overrides
        strategy_config_url = config.DATABASE_URL.replace('/forex', '/strategy_config')
        self.strategy_db = DatabaseManager(strategy_config_url)

        self.correlation_analyzer = CorrelationAnalyzer(min_sample_size, min_confidence)
        self.direction_analyzer = DirectionAnalyzer(min_sample_size, min_confidence)
        self.regime_analyzer = RegimeAnalyzer(min_sample_size, min_confidence)

        self.sql_generator = SQLGenerator()
        self.report_generator = ReportGenerator()

    def run(self) -> Dict[str, Any]:
        """
        Run the full optimization analysis.

        Returns:
            Dictionary with all results and recommendations
        """
        self.logger.info(f"Starting unified parameter optimization for {self.days} days...")

        results = {
            'generated_at': datetime.now().isoformat(),
            'days': self.days,
            'min_sample_size': self.min_sample_size,
            'min_confidence': self.min_confidence,
        }

        # Step 1: Collect data
        self.logger.info("Step 1/4: Collecting data...")
        trade_df = self.trade_collector.collect(self.days, self.epics)
        rejection_df = self.rejection_collector.collect(self.days, self.epics)
        market_intel_df = self.market_intel_collector.collect(self.days)

        self.logger.info(f"  Trades: {len(trade_df)}")
        self.logger.info(f"  Rejections: {len(rejection_df)}")
        self.logger.info(f"  Market intel snapshots: {len(market_intel_df)}")

        # Filter to only enabled epics (when no specific epics provided)
        if not self.epics:
            enabled_epics = self._get_enabled_epics()
            if enabled_epics:
                self.logger.info(f"  Filtering to {len(enabled_epics)} enabled epics")
                if not rejection_df.empty:
                    before_count = len(rejection_df)
                    rejection_df = rejection_df[rejection_df['epic'].isin(enabled_epics)]
                    self.logger.info(f"    Rejections: {before_count} -> {len(rejection_df)}")
                if not trade_df.empty:
                    before_count = len(trade_df)
                    trade_df = trade_df[trade_df['epic'].isin(enabled_epics)]
                    self.logger.info(f"    Trades: {before_count} -> {len(trade_df)}")

        if trade_df.empty and rejection_df.empty:
            self.logger.warning("No data found for analysis!")
            return results

        # Get trade summaries by epic
        trade_summaries = {}
        if not trade_df.empty:
            summary_df = self.trade_collector.get_summary_by_epic(trade_df)
            trade_summaries = summary_df.set_index('epic').to_dict('index')

        results['trade_summaries'] = trade_summaries

        # Build rejection stats for reporting
        rejection_stats = {}
        if not rejection_df.empty:
            rejection_stats = {
                'total_rejections': len(rejection_df),
                'epics_count': rejection_df['epic'].nunique(),
                'epics': rejection_df['epic'].unique().tolist()
            }
        results['rejection_stats'] = rejection_stats

        # Step 2: Load current configuration
        self.logger.info("Step 2/4: Loading current configuration...")
        current_config = self._load_current_config()
        results['current_config'] = current_config

        # Step 3: Analyze and generate recommendations
        self.logger.info("Step 3/4: Analyzing parameters...")

        # Correlation analysis (main recommendations)
        recommendations = self.correlation_analyzer.analyze_all_parameters(
            trade_df, rejection_df, current_config
        )
        results['recommendations'] = recommendations
        self.logger.info(f"  Generated {len(recommendations)} parameter recommendations")

        # Direction analysis (if requested)
        direction_recs = []
        direction_perfs = {}
        if self.include_direction_analysis:
            self.logger.info("  Running direction analysis...")
            direction_perfs = self.direction_analyzer.analyze_direction_performance(
                trade_df, rejection_df
            )
            direction_recs = self.direction_analyzer.generate_direction_recommendations(
                direction_perfs, rejection_df, current_config
            )
            self.logger.info(f"  Generated {len(direction_recs)} direction-specific recommendations")

        results['direction_performance'] = direction_perfs
        results['direction_recommendations'] = direction_recs

        # Regime analysis (if requested)
        regime_recs = []
        regime_perfs = {}
        if self.include_regime_analysis:
            self.logger.info("  Running regime analysis...")
            regime_perfs = self.regime_analyzer.analyze_regime_performance(
                trade_df, market_intel_df
            )
            regime_recs = self.regime_analyzer.generate_regime_filter_recommendations(
                regime_perfs, current_config
            )
            self.logger.info(f"  Generated {len(regime_recs)} regime filter recommendations")

        results['regime_performance'] = regime_perfs
        results['regime_recommendations'] = regime_recs

        # Step 4: Generate output
        self.logger.info("Step 4/4: Generating output...")

        # Generate SQL
        sql_by_epic = self.sql_generator.generate_update_sql(
            recommendations, direction_recs, regime_recs, self.min_confidence
        )
        results['sql'] = sql_by_epic

        # Generate report
        report = self.report_generator.generate_full_report(
            days=self.days,
            trade_summaries=trade_summaries,
            direction_perfs=direction_perfs,
            regime_perfs=regime_perfs,
            recommendations=recommendations,
            direction_recs=direction_recs,
            regime_recs=regime_recs,
            sql_by_epic=sql_by_epic,
            min_confidence=self.min_confidence,
            rejection_stats=rejection_stats
        )
        results['report'] = report

        return results

    def _load_current_config(self) -> Dict[str, Dict[str, Any]]:
        """Load current per-epic configuration from database"""
        config = {}

        try:
            # Load ALL configurable parameters from pair_overrides
            query = """
            SELECT
                epic,
                -- SL/TP parameters
                CAST(fixed_stop_loss_pips AS FLOAT) as fixed_stop_loss_pips,
                CAST(fixed_take_profit_pips AS FLOAT) as fixed_take_profit_pips,
                sl_buffer_pips,

                -- Confidence parameters
                CAST(min_confidence AS FLOAT) as min_confidence,
                CAST(max_confidence AS FLOAT) as max_confidence,
                CAST(min_confidence_bull AS FLOAT) as min_confidence_bull,
                CAST(min_confidence_bear AS FLOAT) as min_confidence_bear,

                -- Volume parameters
                CAST(min_volume_ratio AS FLOAT) as min_volume_ratio,
                CAST(min_volume_ratio_bull AS FLOAT) as min_volume_ratio_bull,
                CAST(min_volume_ratio_bear AS FLOAT) as min_volume_ratio_bear,

                -- Fib pullback parameters
                CAST(fib_pullback_min_bull AS FLOAT) as fib_pullback_min_bull,
                CAST(fib_pullback_min_bear AS FLOAT) as fib_pullback_min_bear,
                CAST(fib_pullback_max_bull AS FLOAT) as fib_pullback_max_bull,
                CAST(fib_pullback_max_bear AS FLOAT) as fib_pullback_max_bear,

                -- Momentum parameters
                CAST(momentum_min_depth_bull AS FLOAT) as momentum_min_depth_bull,
                CAST(momentum_min_depth_bear AS FLOAT) as momentum_min_depth_bear,

                -- Swing structure parameters
                CAST(min_swing_atr_multiplier AS FLOAT) as min_swing_atr_multiplier,
                swing_lookback_bars,

                -- Filter toggles
                direction_overrides_enabled,
                macd_filter_enabled,
                smc_conflict_tolerance,

                -- Session parameters
                allow_asian_session,

                -- Confidence adjustments
                CAST(high_volume_confidence AS FLOAT) as high_volume_confidence,
                CAST(low_atr_confidence AS FLOAT) as low_atr_confidence,
                CAST(high_atr_confidence AS FLOAT) as high_atr_confidence,
                CAST(near_ema_confidence AS FLOAT) as near_ema_confidence,
                CAST(far_ema_confidence AS FLOAT) as far_ema_confidence,

                -- Metadata
                change_reason,
                updated_at
            FROM smc_simple_pair_overrides
            WHERE is_enabled = TRUE
            """

            result = self.strategy_db.execute_query(query)

            if result is not None and not result.empty:
                for _, row in result.iterrows():
                    epic = row['epic']
                    config[epic] = row.to_dict()

                    # Also add short name mapping
                    for short, full in self.EPIC_MAPPING.items():
                        if full == epic:
                            config[short] = row.to_dict()

        except Exception as e:
            self.logger.error(f"Failed to load current config: {e}")

        return config

    def _get_enabled_epics(self) -> List[str]:
        """Get list of enabled epics from strategy_config database"""
        try:
            query = """
            SELECT epic FROM smc_simple_pair_overrides
            WHERE is_enabled = TRUE
            """
            result = self.strategy_db.execute_query(query)
            if result is not None and not result.empty:
                return result['epic'].tolist()
        except Exception as e:
            self.logger.warning(f"Failed to load enabled epics: {e}")
        return []

    def print_report(self, results: Dict[str, Any]) -> None:
        """Print the report to console"""
        if 'report' in results:
            print(results['report'])

    def export_sql(self, results: Dict[str, Any], filepath: str) -> None:
        """Export SQL to file"""
        if 'sql' not in results:
            self.logger.error("No SQL to export")
            return

        full_sql = self.sql_generator.generate_full_script(results['sql'])

        with open(filepath, 'w') as f:
            f.write(full_sql)

        self.logger.info(f"SQL exported to {filepath}")

    def export_json(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results as JSON"""
        json_report = self.report_generator.generate_json_report(
            trade_summaries=results.get('trade_summaries', {}),
            recommendations=results.get('recommendations', []),
            direction_recs=results.get('direction_recommendations', []),
            regime_recs=results.get('regime_recommendations', [])
        )

        with open(filepath, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)

        self.logger.info(f"JSON exported to {filepath}")

    def apply_changes(self, results: Dict[str, Any]) -> bool:
        """
        Apply recommended changes to database (with confirmation).

        Returns:
            True if changes were applied, False otherwise
        """
        if 'sql' not in results or not results['sql']:
            self.logger.warning("No changes to apply")
            return False

        # Print summary of changes
        print("\n" + "=" * 80)
        print("CHANGES TO BE APPLIED:")
        print("=" * 80)

        for epic, sql in results['sql'].items():
            if sql and not sql.startswith('--'):
                print(f"\n{epic}:")
                print(sql[:500] + "..." if len(sql) > 500 else sql)

        print("\n" + "=" * 80)
        print("WARNING: This will modify the smc_simple_pair_overrides table!")
        print("=" * 80)

        # Confirm
        response = input("\nDo you want to apply these changes? (yes/no): ")

        if response.lower() != 'yes':
            print("Changes NOT applied.")
            return False

        # Apply changes
        try:
            full_sql = self.sql_generator.generate_full_script(results['sql'], include_header=False)

            # Remove ROLLBACK, add COMMIT
            full_sql = full_sql.replace('ROLLBACK;', 'COMMIT;')

            self.db_manager.execute_query(full_sql)
            print("\nChanges applied successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply changes: {e}")
            return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Unified Parameter Optimizer for SMC Simple Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic dry run
    python unified_parameter_optimizer.py

    # Full analysis with all options
    python unified_parameter_optimizer.py --days 30 --direction-analysis --include-regime-analysis -v

    # Export SQL to file
    python unified_parameter_optimizer.py --export-sql /tmp/optimization.sql

    # Specific epic analysis
    python unified_parameter_optimizer.py --epic EURUSD GBPUSD

    # Apply changes (with confirmation prompt)
    python unified_parameter_optimizer.py --apply
        """
    )

    # Data options
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of data to analyze (default: 30)')
    parser.add_argument('--epic', nargs='+', type=str,
                        help='Specific epic(s) to analyze (e.g., EURUSD GBPUSD)')

    # Analysis options
    parser.add_argument('--min-sample-size', type=int, default=20,
                        help='Minimum sample size for recommendations (default: 20)')
    parser.add_argument('--min-confidence', type=float, default=0.70,
                        help='Minimum confidence level (default: 0.70)')
    parser.add_argument('--direction-analysis', action='store_true',
                        help='Include BULL vs BEAR direction analysis')
    parser.add_argument('--include-regime-analysis', action='store_true',
                        help='Include market regime correlation analysis')

    # Output options
    parser.add_argument('--export-sql', type=str, metavar='FILE',
                        help='Export SQL statements to file')
    parser.add_argument('--export-json', type=str, metavar='FILE',
                        help='Export results as JSON')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    # Apply option
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes to database (prompts for confirmation)')

    args = parser.parse_args()

    # Create optimizer
    optimizer = UnifiedParameterOptimizer(
        days=args.days,
        min_sample_size=args.min_sample_size,
        min_confidence=args.min_confidence,
        epics=args.epic,
        include_direction_analysis=args.direction_analysis,
        include_regime_analysis=args.include_regime_analysis,
        verbose=args.verbose
    )

    # Run analysis
    results = optimizer.run()

    # Print report
    optimizer.print_report(results)

    # Export if requested
    if args.export_sql:
        optimizer.export_sql(results, args.export_sql)

    if args.export_json:
        optimizer.export_json(results, args.export_json)

    # Apply if requested
    if args.apply:
        optimizer.apply_changes(results)


if __name__ == '__main__':
    main()
