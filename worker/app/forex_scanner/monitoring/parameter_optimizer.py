#!/usr/bin/env python3
"""
SMC Simple Parameter Optimizer

Fetches rejection outcome analysis recommendations and generates or applies
database configuration updates to optimize strategy parameters.

Features:
- Fetches recommendations from FastAPI rejection outcome API
- Generates SQL updates for smc_simple_pair_overrides table
- Supports dry-run mode (show changes without applying)
- Supports auto-apply mode (apply changes directly to database)
- Logs all changes with audit trail

Usage:
    # Show recommendations without making changes
    docker exec -it task-worker python /app/forex_scanner/monitoring/parameter_optimizer.py --days 30

    # Apply changes to database
    docker exec -it task-worker python /app/forex_scanner/monitoring/parameter_optimizer.py --days 30 --apply

    # Export SQL file only
    docker exec -it task-worker python /app/forex_scanner/monitoring/parameter_optimizer.py --days 30 --export-sql

Created: 2025-12-31
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal

import requests
import psycopg2
import psycopg2.extras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping from rejection stages to adjustable parameters
STAGE_PARAMETER_MAPPINGS = {
    'TIER1_EMA': {
        'global_params': ['min_distance_from_ema_pips', 'ema_buffer_pips'],
        'pair_params': ['near_ema_confidence', 'far_ema_confidence'],
        'relax_action': {'ema_buffer_pips': -0.5, 'min_distance_from_ema_pips': -0.5},
        'tighten_action': {'ema_buffer_pips': +0.5, 'min_distance_from_ema_pips': +0.5},
    },
    'TIER2_SWING': {
        'global_params': ['min_body_percentage', 'min_breakout_atr_ratio'],
        'pair_params': ['parameter_overrides.MIN_BODY_PERCENTAGE', 'parameter_overrides.MIN_BREAKOUT_ATR_RATIO'],
        'relax_action': {'min_body_percentage': -0.05, 'min_breakout_atr_ratio': -0.05},
        'tighten_action': {'min_body_percentage': +0.05, 'min_breakout_atr_ratio': +0.05},
    },
    'TIER3_PULLBACK': {
        'global_params': ['fib_pullback_min', 'fib_pullback_max', 'momentum_min_depth'],
        'pair_params': ['parameter_overrides.FIB_PULLBACK_MIN', 'parameter_overrides.MOMENTUM_MIN_DEPTH'],
        'relax_action': {'fib_pullback_min': -0.02, 'fib_pullback_max': +0.05, 'momentum_min_depth': -0.1},
        'tighten_action': {'fib_pullback_min': +0.02, 'fib_pullback_max': -0.05, 'momentum_min_depth': +0.1},
    },
    'CONFIDENCE': {
        'global_params': ['min_confidence_threshold'],
        'pair_params': ['min_confidence'],
        'relax_action': {'min_confidence_threshold': -0.02},
        'tighten_action': {'min_confidence_threshold': +0.02},
    },
    'VOLUME_LOW': {
        'global_params': ['min_volume_ratio'],
        'pair_params': ['min_volume_ratio'],
        'relax_action': {'min_volume_ratio': -0.05},
        'tighten_action': {'min_volume_ratio': +0.05},
    },
    'MACD_MISALIGNED': {
        'global_params': ['macd_alignment_filter_enabled'],
        'pair_params': ['macd_filter_enabled'],
        'relax_action': {'macd_filter_enabled': False},  # Disable if rejecting too many winners
        'tighten_action': {'macd_filter_enabled': True},
    },
    'RISK_RR': {
        'global_params': ['min_rr_ratio'],
        'pair_params': [],
        'relax_action': {'min_rr_ratio': -0.1},
        'tighten_action': {'min_rr_ratio': +0.1},
    },
}


class ParameterOptimizer:
    """Fetches recommendations and generates/applies config updates."""

    def __init__(
        self,
        fastapi_url: str = None,
        database_url: str = None,
        api_key: str = None
    ):
        self.fastapi_url = fastapi_url or os.getenv(
            'FASTAPI_URL', 'http://fastapi-dev:8000'
        )
        self.database_url = database_url or os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )
        self.api_key = api_key or os.getenv(
            'FASTAPI_API_KEY', '436abe054a074894a0517e5172f0e5b6'
        )
        self.headers = {
            'X-APIM-Gateway': 'verified',
            'X-API-KEY': self.api_key
        }

    def fetch_recommendations(self, days: int = 30) -> Dict[str, Any]:
        """Fetch parameter suggestions from the rejection outcome API."""
        try:
            url = f"{self.fastapi_url}/api/rejection-outcomes/parameter-suggestions"
            response = requests.get(
                url,
                params={'days': days},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch recommendations: {e}")
            return {}

    def fetch_stage_metrics(self, days: int = 30) -> List[Dict]:
        """Fetch win rate by stage from the API."""
        try:
            url = f"{self.fastapi_url}/api/rejection-outcomes/win-rate-by-stage"
            response = requests.get(
                url,
                params={'days': days},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch stage metrics: {e}")
            return []

    def fetch_pair_metrics(self, days: int = 30) -> List[Dict]:
        """Fetch win rate by pair from the API."""
        try:
            url = f"{self.fastapi_url}/api/rejection-outcomes/by-pair"
            response = requests.get(
                url,
                params={'days': days},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch pair metrics: {e}")
            return []

    def get_current_config(self) -> Tuple[int, Dict[str, Any], Dict[str, Dict]]:
        """Get current configuration from database."""
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get active global config
                cur.execute("""
                    SELECT * FROM smc_simple_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                global_config = dict(cur.fetchone() or {})
                config_id = global_config.get('id', 0)

                # Get pair overrides
                cur.execute("""
                    SELECT * FROM smc_simple_pair_overrides
                    WHERE config_id = %s
                """, (config_id,))
                pair_rows = cur.fetchall()
                pair_overrides = {row['epic']: dict(row) for row in pair_rows}

                return config_id, global_config, pair_overrides
        finally:
            conn.close()

    def generate_recommendations(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate actionable parameter change recommendations.

        Returns list of changes with:
        - scope: 'global' or 'pair'
        - target: parameter name or (epic, parameter_name)
        - current_value: current value
        - recommended_value: new value
        - reason: why the change is recommended
        - confidence: how confident we are (based on sample size)
        """
        recommendations = []

        # Fetch data
        stage_metrics = self.fetch_stage_metrics(days)
        pair_metrics = self.fetch_pair_metrics(days)
        config_id, global_config, pair_overrides = self.get_current_config()

        if not stage_metrics:
            logger.warning("No stage metrics available")
            return recommendations

        # Analyze each stage
        for stage in stage_metrics:
            stage_name = stage.get('rejection_stage', '')
            win_rate = stage.get('would_be_win_rate', 50)
            total = stage.get('total_analyzed', 0)
            missed_pips = stage.get('missed_profit_pips', 0)
            avoided_pips = stage.get('avoided_loss_pips', 0)

            # Skip stages with too few samples
            if total < 20:
                continue

            # Skip stages we don't have mappings for
            if stage_name not in STAGE_PARAMETER_MAPPINGS:
                continue

            mapping = STAGE_PARAMETER_MAPPINGS[stage_name]

            # Determine action based on win rate
            if win_rate > 60:
                # Too aggressive - relax filters
                action = 'relax'
                action_params = mapping.get('relax_action', {})
                reason = f"{stage_name} rejects {win_rate:.0f}% would-be winners ({total} samples, {missed_pips:.0f} missed pips)"
            elif win_rate < 40:
                # Working well - could tighten slightly
                action = 'tighten'
                action_params = mapping.get('tighten_action', {})
                reason = f"{stage_name} correctly filters {100-win_rate:.0f}% losers ({total} samples, {avoided_pips:.0f} avoided pips)"
            else:
                # Neutral - no change needed
                continue

            # Generate global parameter changes
            for param in mapping.get('global_params', []):
                if param in action_params:
                    current_value = global_config.get(param)
                    if current_value is not None:
                        adjustment = action_params[param]
                        if isinstance(adjustment, bool):
                            new_value = adjustment
                        else:
                            # Convert Decimal to float for arithmetic
                            if isinstance(current_value, Decimal):
                                current_value = float(current_value)
                            new_value = round(current_value + adjustment, 3)

                        recommendations.append({
                            'scope': 'global',
                            'target': param,
                            'current_value': current_value,
                            'recommended_value': new_value,
                            'reason': reason,
                            'action': action,
                            'stage': stage_name,
                            'confidence': min(total / 100, 1.0),  # 0-1 based on sample size
                            'impact_pips': missed_pips if action == 'relax' else avoided_pips,
                        })

        # Analyze pair-specific recommendations
        for pair_info in pair_metrics:
            epic = pair_info.get('epic', '')
            pair = pair_info.get('pair', '')
            win_rate = pair_info.get('would_be_win_rate', 50)
            total = pair_info.get('total_analyzed', 0)
            status = pair_info.get('status', 'NEUTRAL')
            missed_pips = pair_info.get('missed_profit_pips', 0)
            avoided_pips = pair_info.get('avoided_loss_pips', 0)

            if total < 15:
                continue

            current_override = pair_overrides.get(epic, {})

            if status == 'TOO_AGGRESSIVE' and win_rate > 60:
                # Relax confidence threshold for this pair
                current_conf = current_override.get('min_confidence') or global_config.get('min_confidence_threshold', 0.48)
                if isinstance(current_conf, Decimal):
                    current_conf = float(current_conf)
                new_conf = max(0.40, current_conf - 0.03)

                recommendations.append({
                    'scope': 'pair',
                    'target': (epic, 'min_confidence'),
                    'current_value': current_conf,
                    'recommended_value': round(new_conf, 3),
                    'reason': f"{pair} filters reject {win_rate:.0f}% would-be winners ({total} samples)",
                    'action': 'relax',
                    'stage': 'PAIR_SPECIFIC',
                    'confidence': min(total / 50, 1.0),
                    'impact_pips': missed_pips,
                })

            elif status == 'WORKING_WELL' and win_rate < 40:
                # Tighten or add volume filter
                current_vol = current_override.get('min_volume_ratio') or global_config.get('min_volume_ratio', 0.5)
                if isinstance(current_vol, Decimal):
                    current_vol = float(current_vol)
                new_vol = min(0.80, current_vol + 0.05)

                recommendations.append({
                    'scope': 'pair',
                    'target': (epic, 'min_volume_ratio'),
                    'current_value': current_vol,
                    'recommended_value': round(new_vol, 2),
                    'reason': f"{pair} filters are protective ({100-win_rate:.0f}% losers filtered)",
                    'action': 'tighten',
                    'stage': 'PAIR_SPECIFIC',
                    'confidence': min(total / 50, 1.0),
                    'impact_pips': avoided_pips,
                })

        return recommendations

    def generate_sql(self, recommendations: List[Dict]) -> str:
        """Generate SQL statements to apply recommendations."""
        if not recommendations:
            return "-- No recommendations to apply\n"

        sql_lines = [
            "-- SMC Simple Parameter Optimization",
            f"-- Generated: {datetime.now().isoformat()}",
            f"-- Recommendations: {len(recommendations)}",
            "",
            "BEGIN;",
            ""
        ]

        config_id, _, _ = self.get_current_config()

        for rec in recommendations:
            if rec['scope'] == 'global':
                param = rec['target']
                new_value = rec['recommended_value']

                if isinstance(new_value, bool):
                    value_str = 'TRUE' if new_value else 'FALSE'
                elif isinstance(new_value, str):
                    value_str = f"'{new_value}'"
                else:
                    value_str = str(new_value)

                sql_lines.append(f"-- {rec['reason']}")
                sql_lines.append(f"-- Action: {rec['action']} (confidence: {rec['confidence']:.0%})")
                sql_lines.append(f"UPDATE smc_simple_global_config SET")
                sql_lines.append(f"    {param} = {value_str},")
                sql_lines.append(f"    strategy_status = 'Auto-optimized {param}: {rec['current_value']} -> {new_value}',")
                sql_lines.append(f"    updated_at = NOW()")
                sql_lines.append(f"WHERE id = {config_id};")
                sql_lines.append("")

            elif rec['scope'] == 'pair':
                epic, param = rec['target']
                new_value = rec['recommended_value']

                if isinstance(new_value, bool):
                    value_str = 'TRUE' if new_value else 'FALSE'
                elif isinstance(new_value, str):
                    value_str = f"'{new_value}'"
                else:
                    value_str = str(new_value)

                sql_lines.append(f"-- {rec['reason']}")
                sql_lines.append(f"-- Action: {rec['action']} for {epic}")
                reason_truncated = rec['reason'][:100]
                sql_lines.append(f"INSERT INTO smc_simple_pair_overrides (config_id, epic, {param}, change_reason)")
                sql_lines.append(f"VALUES ({config_id}, '{epic}', {value_str}, 'Auto-optimized: {reason_truncated}')")
                sql_lines.append(f"ON CONFLICT (config_id, epic) DO UPDATE SET")
                sql_lines.append(f"    {param} = {value_str},")
                sql_lines.append(f"    change_reason = 'Auto-optimized: {reason_truncated}',")
                sql_lines.append(f"    updated_at = NOW();")
                sql_lines.append("")

        sql_lines.append("COMMIT;")
        return "\n".join(sql_lines)

    def apply_recommendations(self, recommendations: List[Dict]) -> bool:
        """Apply recommendations directly to the database."""
        if not recommendations:
            logger.info("No recommendations to apply")
            return True

        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                config_id, _, _ = self.get_current_config()

                for rec in recommendations:
                    if rec['scope'] == 'global':
                        param = rec['target']
                        new_value = rec['recommended_value']

                        cur.execute(f"""
                            UPDATE smc_simple_global_config SET
                                {param} = %s,
                                strategy_status = %s,
                                updated_at = NOW()
                            WHERE id = %s
                        """, (
                            new_value,
                            f"Auto-optimized {param}: {rec['current_value']} -> {new_value}",
                            config_id
                        ))
                        logger.info(f"Updated global.{param}: {rec['current_value']} -> {new_value}")

                    elif rec['scope'] == 'pair':
                        epic, param = rec['target']
                        new_value = rec['recommended_value']

                        cur.execute(f"""
                            INSERT INTO smc_simple_pair_overrides (config_id, epic, {param}, change_reason)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (config_id, epic) DO UPDATE SET
                                {param} = EXCLUDED.{param},
                                change_reason = EXCLUDED.change_reason,
                                updated_at = NOW()
                        """, (
                            config_id,
                            epic,
                            new_value,
                            f"Auto-optimized: {rec['reason'][:100]}"
                        ))
                        logger.info(f"Updated {epic}.{param}: {rec['current_value']} -> {new_value}")

                conn.commit()
                logger.info(f"Applied {len(recommendations)} recommendations successfully")
                return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply recommendations: {e}")
            return False
        finally:
            conn.close()

    def print_report(self, recommendations: List[Dict], detailed: bool = True):
        """Print a formatted report of recommendations."""
        print("\n" + "=" * 80)
        print("SMC SIMPLE PARAMETER OPTIMIZATION REPORT")
        print("=" * 80)

        if not recommendations:
            print("\nNo recommendations at this time.")
            print("This could mean:")
            print("  - Filters are well-balanced")
            print("  - Insufficient data (need more rejection outcomes)")
            print("  - Run the rejection_outcome_analyzer first")
            return

        # Group by action
        relax_recs = [r for r in recommendations if r['action'] == 'relax']
        tighten_recs = [r for r in recommendations if r['action'] == 'tighten']

        if relax_recs:
            print(f"\n{'─' * 40}")
            print("RELAX FILTERS (Capturing more winners)")
            print(f"{'─' * 40}")
            total_missed = sum(r.get('impact_pips', 0) for r in relax_recs)
            print(f"Potential recovery: {total_missed:.0f} pips\n")

            for rec in relax_recs:
                if rec['scope'] == 'global':
                    print(f"  • {rec['target']}: {rec['current_value']} → {rec['recommended_value']}")
                else:
                    epic, param = rec['target']
                    print(f"  • {epic}.{param}: {rec['current_value']} → {rec['recommended_value']}")
                if detailed:
                    print(f"    Reason: {rec['reason']}")
                    print(f"    Confidence: {rec['confidence']:.0%}")
                    print()

        if tighten_recs:
            print(f"\n{'─' * 40}")
            print("TIGHTEN FILTERS (Avoiding more losers)")
            print(f"{'─' * 40}")
            total_avoided = sum(r.get('impact_pips', 0) for r in tighten_recs)
            print(f"Potential savings: {total_avoided:.0f} pips\n")

            for rec in tighten_recs:
                if rec['scope'] == 'global':
                    print(f"  • {rec['target']}: {rec['current_value']} → {rec['recommended_value']}")
                else:
                    epic, param = rec['target']
                    print(f"  • {epic}.{param}: {rec['current_value']} → {rec['recommended_value']}")
                if detailed:
                    print(f"    Reason: {rec['reason']}")
                    print(f"    Confidence: {rec['confidence']:.0%}")
                    print()

        print("=" * 80)
        print(f"Total recommendations: {len(recommendations)}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Optimize SMC Simple strategy parameters based on rejection outcome analysis'
    )
    parser.add_argument(
        '--days', type=int, default=30,
        help='Number of days to analyze (default: 30)'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Apply recommendations to database'
    )
    parser.add_argument(
        '--export-sql', action='store_true',
        help='Export SQL file instead of applying'
    )
    parser.add_argument(
        '--output', type=str, default='parameter_updates.sql',
        help='SQL output file path (default: parameter_updates.sql)'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.5,
        help='Minimum confidence threshold for applying recommendations (default: 0.5)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    optimizer = ParameterOptimizer()

    # Generate recommendations
    print(f"\nAnalyzing rejection outcomes for the last {args.days} days...")
    recommendations = optimizer.generate_recommendations(args.days)

    # Filter by confidence
    if args.min_confidence > 0:
        original_count = len(recommendations)
        recommendations = [r for r in recommendations if r['confidence'] >= args.min_confidence]
        if original_count != len(recommendations):
            print(f"Filtered {original_count - len(recommendations)} low-confidence recommendations")

    # Print report
    optimizer.print_report(recommendations, detailed=not args.quiet)

    if not recommendations:
        return

    if args.export_sql:
        # Export SQL file
        sql = optimizer.generate_sql(recommendations)
        with open(args.output, 'w') as f:
            f.write(sql)
        print(f"\nSQL exported to: {args.output}")

    elif args.apply:
        # Apply directly
        print("\nApplying recommendations to database...")
        if optimizer.apply_recommendations(recommendations):
            print("✅ Recommendations applied successfully!")
            print("   The config service will pick up changes on next cache refresh (2 min)")
        else:
            print("❌ Failed to apply recommendations")
            sys.exit(1)

    else:
        # Dry run
        print("\nThis was a DRY RUN. No changes were made.")
        print("To apply changes, run with --apply flag")
        print("To export SQL, run with --export-sql flag")


if __name__ == '__main__':
    main()
