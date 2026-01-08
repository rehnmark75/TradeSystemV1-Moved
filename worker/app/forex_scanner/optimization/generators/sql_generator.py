# optimization/generators/sql_generator.py
"""
SQL generator for unified parameter optimizer.
Generates UPDATE statements for smc_simple_pair_overrides table.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..analyzers.correlation_analyzer import OptimizationRecommendation
from ..analyzers.direction_analyzer import DirectionRecommendation
from ..analyzers.regime_analyzer import RegimeFilterRecommendation


class SQLGenerator:
    """Generates SQL UPDATE statements for parameter changes"""

    # Epic mapping for database table
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

    # Parameters that can be set per-pair in smc_simple_pair_overrides
    PER_PAIR_PARAMS = {
        'min_confidence', 'max_confidence', 'min_volume_ratio',
        'fixed_stop_loss_pips', 'fixed_take_profit_pips',
        'sl_buffer_pips', 'macd_filter_enabled',
        'fib_pullback_min_bull', 'fib_pullback_min_bear',
        'fib_pullback_max_bull', 'fib_pullback_max_bear',
        'momentum_min_depth_bull', 'momentum_min_depth_bear',
        'min_volume_ratio_bull', 'min_volume_ratio_bear',
        'min_confidence_bull', 'min_confidence_bear',
        'direction_overrides_enabled',
        # New swing/filter overrides
        'min_swing_atr_multiplier', 'swing_lookback_bars',
        'smc_conflict_tolerance',
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_update_sql(
        self,
        recommendations: List[OptimizationRecommendation],
        direction_recs: List[DirectionRecommendation] = None,
        regime_recs: List[RegimeFilterRecommendation] = None,
        min_confidence: float = 0.70
    ) -> Dict[str, str]:
        """
        Generate SQL UPDATE statements grouped by epic.

        Args:
            recommendations: Correlation-based recommendations
            direction_recs: Direction-specific recommendations
            regime_recs: Regime filter recommendations
            min_confidence: Minimum confidence to include

        Returns:
            Dict[epic, sql_statement]
        """
        updates_by_epic = {}

        # Collect all updates by epic
        for rec in recommendations:
            if rec.confidence < min_confidence:
                continue

            epic = self._normalize_epic(rec.epic)
            if epic not in updates_by_epic:
                updates_by_epic[epic] = {'params': {}, 'notes': []}

            updates_by_epic[epic]['params'][rec.parameter] = rec.recommended_value
            updates_by_epic[epic]['notes'].append(
                f"{rec.parameter}: {rec.current_value} -> {rec.recommended_value} ({rec.confidence:.0%})"
            )

        # Add direction recommendations
        if direction_recs:
            for rec in direction_recs:
                if rec.confidence < min_confidence:
                    continue

                epic = self._normalize_epic(rec.epic)
                if epic not in updates_by_epic:
                    updates_by_epic[epic] = {'params': {}, 'notes': []}

                updates_by_epic[epic]['params'][rec.parameter] = rec.recommended_value
                updates_by_epic[epic]['notes'].append(
                    f"{rec.parameter}: {rec.current_value} -> {rec.recommended_value} ({rec.direction})"
                )

        # Generate SQL for each epic
        sql_statements = {}
        for epic, data in updates_by_epic.items():
            sql_statements[epic] = self._build_update_statement(
                epic, data['params'], data['notes']
            )

        # Add regime filter recommendations as comments
        if regime_recs:
            for rec in regime_recs:
                if rec.confidence < min_confidence:
                    continue

                epic = self._normalize_epic(rec.epic)
                if epic not in sql_statements:
                    sql_statements[epic] = ""

                regime_sql = self._build_regime_comment(rec)
                sql_statements[epic] += f"\n\n{regime_sql}"

        return sql_statements

    def _normalize_epic(self, epic: str) -> str:
        """Normalize epic to full format"""
        if epic in self.EPIC_MAPPING:
            return self.EPIC_MAPPING[epic]
        if epic.startswith('CS.D.'):
            return epic
        # Try to match partial name
        for short, full in self.EPIC_MAPPING.items():
            if short in epic:
                return full
        return epic

    def _build_update_statement(
        self,
        epic: str,
        params: Dict[str, Any],
        notes: List[str]
    ) -> str:
        """Build UPDATE statement for an epic"""
        if not params:
            return ""

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Separate per-pair params from global params
        per_pair_params = {}
        global_params = {}

        for param, value in params.items():
            if param in self.PER_PAIR_PARAMS:
                per_pair_params[param] = value
            else:
                global_params[param] = value

        sql_parts = []

        # Build per-pair UPDATE statement
        if per_pair_params:
            set_parts = []
            for param, value in per_pair_params.items():
                if isinstance(value, bool):
                    set_parts.append(f"    {param} = {str(value).upper()}")
                elif isinstance(value, str):
                    set_parts.append(f"    {param} = '{value}'")
                elif value is None:
                    set_parts.append(f"    {param} = NULL")
                else:
                    set_parts.append(f"    {param} = {value}")

            # Add change reason
            reason = f"Unified optimizer: {len(per_pair_params)} changes on {timestamp}"
            set_parts.append(f"    change_reason = '{reason}'")
            set_parts.append(f"    updated_at = NOW()")

            set_clause = ",\n".join(set_parts)

            per_pair_notes = [n for n in notes if any(p in n for p in per_pair_params.keys())]

            sql_parts.append(f"""-- Optimizations for {epic}
-- Changes: {'; '.join(per_pair_notes) if per_pair_notes else '; '.join(notes)}
UPDATE smc_simple_pair_overrides SET
{set_clause}
WHERE epic = '{epic}';""")

        # Build global config UPDATE statement (as comment/suggestion)
        if global_params:
            global_notes = [n for n in notes if any(p in n for p in global_params.keys())]
            global_sql_parts = []
            for param, value in global_params.items():
                if isinstance(value, bool):
                    global_sql_parts.append(f"    {param} = {str(value).upper()}")
                elif isinstance(value, str):
                    global_sql_parts.append(f"    {param} = '{value}'")
                else:
                    global_sql_parts.append(f"    {param} = {value}")

            global_set_clause = ",\n--     ".join(global_sql_parts)

            sql_parts.append(f"""
-- GLOBAL CONFIG SUGGESTION for {epic}:
-- These parameters are global (affect all pairs). Consider adding per-pair columns.
-- Changes: {'; '.join(global_notes)}
-- To apply globally:
-- UPDATE smc_simple_global_config SET
--     {global_set_clause},
--     change_reason = 'Unified optimizer suggestion for {epic}',
--     updated_at = NOW()
-- WHERE is_active = TRUE;""")

        return "\n".join(sql_parts)

    def _build_regime_comment(self, rec: RegimeFilterRecommendation) -> str:
        """Build comment for regime filter recommendation"""
        epic = self._normalize_epic(rec.epic)
        regimes = ', '.join(rec.block_regimes)

        comment = f"""-- REGIME FILTER RECOMMENDATION for {epic}
-- Block regimes: {regimes}
-- Reason: {rec.reason}
-- Confidence: {rec.confidence:.0%} | Sample: {rec.sample_size} trades | Impact: {rec.estimated_impact_pips:.0f} pips
--
-- To implement:
-- Option 1: Add blocked_regimes column to smc_simple_pair_overrides
--   ALTER TABLE smc_simple_pair_overrides ADD COLUMN blocked_regimes TEXT[];
--   UPDATE smc_simple_pair_overrides
--   SET blocked_regimes = ARRAY['{regimes.replace(", ", "', '")}']
--   WHERE epic = '{epic}';
--
-- Option 2: Add check in scanner logic before signal generation
--   if market_regime in ['{"', '".join(rec.block_regimes)}']:
--       return None  # Skip signal in poor-performing regime"""

        return comment

    def generate_full_script(
        self,
        sql_by_epic: Dict[str, str],
        include_header: bool = True
    ) -> str:
        """
        Generate complete SQL script with all updates.

        Args:
            sql_by_epic: SQL statements by epic
            include_header: Include transaction header/footer

        Returns:
            Complete SQL script
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        parts = []

        if include_header:
            parts.append(f"""-- ═══════════════════════════════════════════════════════════════════════════════
-- UNIFIED PARAMETER OPTIMIZATION SQL
-- Generated: {timestamp}
-- Database: strategy_config
-- Table: smc_simple_pair_overrides
-- ═══════════════════════════════════════════════════════════════════════════════

-- IMPORTANT: Review all changes before executing!
-- Run with: docker exec postgres psql -U postgres -d strategy_config -f <file>

BEGIN;
""")

        # Add each epic's SQL
        for epic, sql in sorted(sql_by_epic.items()):
            if sql.strip():
                parts.append(sql)
                parts.append("")

        if include_header:
            parts.append("""-- Verify changes
SELECT epic,
       fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, min_volume_ratio,
       direction_overrides_enabled,
       change_reason,
       updated_at
FROM smc_simple_pair_overrides
ORDER BY updated_at DESC
LIMIT 10;

-- Uncomment to commit changes:
-- COMMIT;

-- Rollback if needed:
ROLLBACK;
""")

        return "\n".join(parts)

    def generate_insert_audit(
        self,
        recommendations: List[OptimizationRecommendation],
        direction_recs: List[DirectionRecommendation] = None
    ) -> str:
        """Generate INSERT statements for audit table"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        parts = []

        parts.append(f"""-- Audit log entries
-- Table: smc_simple_config_audit
""")

        for rec in recommendations:
            epic = self._normalize_epic(rec.epic)
            parts.append(f"""INSERT INTO smc_simple_config_audit
    (epic, parameter_name, old_value, new_value, change_reason, changed_by, created_at)
VALUES
    ('{epic}', '{rec.parameter}', '{rec.current_value}', '{rec.recommended_value}',
     'Unified optimizer recommendation (confidence: {rec.confidence:.0%})',
     'unified_optimizer', '{timestamp}');""")

        if direction_recs:
            for rec in direction_recs:
                epic = self._normalize_epic(rec.epic)
                parts.append(f"""INSERT INTO smc_simple_config_audit
    (epic, parameter_name, old_value, new_value, change_reason, changed_by, created_at)
VALUES
    ('{epic}', '{rec.parameter}', '{rec.current_value}', '{rec.recommended_value}',
     'Direction-specific: {rec.direction} ({rec.confidence:.0%})',
     'unified_optimizer', '{timestamp}');""")

        return "\n".join(parts)
