"""
Rejection Outcome Analysis Service

Provides data access methods for SMC rejection outcome analysis.
Used by the FastAPI router to serve data to the Unified Analytics dashboard.

Created: 2025-12-28
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


class RejectionOutcomeService:
    """Service for SMC rejection outcome analysis data"""

    def __init__(self, db: Session):
        self.db = db

    def get_outcome_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get high-level outcome summary statistics.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with summary metrics
        """
        query = text("""
        SELECT
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as losers,
            COUNT(CASE WHEN outcome = 'STILL_OPEN' THEN 1 END) as still_open,
            COUNT(CASE WHEN outcome = 'INSUFFICIENT_DATA' THEN 1 END) as insufficient_data,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as total_missed_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_potential_pips,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe_pips,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae_pips,
            ROUND(AVG(time_to_outcome_minutes)::numeric, 0) as avg_time_to_outcome_mins
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            row = result.fetchone()

            if row:
                return {
                    'total_analyzed': row.total_analyzed or 0,
                    'winners': row.winners or 0,
                    'losers': row.losers or 0,
                    'still_open': row.still_open or 0,
                    'insufficient_data': row.insufficient_data or 0,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'total_missed_pips': float(row.total_missed_pips) if row.total_missed_pips else 0.0,
                    'avoided_loss_pips': float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0,
                    'net_potential_pips': float(row.net_potential_pips) if row.net_potential_pips else 0.0,
                    'avg_mfe_pips': float(row.avg_mfe_pips) if row.avg_mfe_pips else 0.0,
                    'avg_mae_pips': float(row.avg_mae_pips) if row.avg_mae_pips else 0.0,
                    'avg_time_to_outcome_mins': int(row.avg_time_to_outcome_mins) if row.avg_time_to_outcome_mins else 0,
                    'days': days
                }
            return {'total_analyzed': 0, 'days': days}

        except Exception as e:
            logger.error(f"Error getting outcome summary: {e}")
            return {'total_analyzed': 0, 'error': str(e), 'days': days}

    def get_win_rate_by_stage(self, days: int = 30) -> List[Dict]:
        """
        Get win rate breakdown by rejection stage.

        Args:
            days: Number of days to include

        Returns:
            List of dictionaries with stage-level metrics
        """
        query = text("""
        SELECT
            rejection_stage,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            COUNT(CASE WHEN outcome = 'STILL_OPEN' THEN 1 END) as still_open,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_pips,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe_pips,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae_pips,
            ROUND(AVG(time_to_outcome_minutes)::numeric, 0) as avg_time_to_outcome_mins
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
        GROUP BY rejection_stage
        ORDER BY total_analyzed DESC
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            rows = result.fetchall()

            return [
                {
                    'rejection_stage': row.rejection_stage,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'still_open': row.still_open,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'missed_profit_pips': float(row.missed_profit_pips) if row.missed_profit_pips else 0.0,
                    'avoided_loss_pips': float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0,
                    'net_pips': float(row.net_pips) if row.net_pips else 0.0,
                    'avg_mfe_pips': float(row.avg_mfe_pips) if row.avg_mfe_pips else 0.0,
                    'avg_mae_pips': float(row.avg_mae_pips) if row.avg_mae_pips else 0.0,
                    'avg_time_to_outcome_mins': int(row.avg_time_to_outcome_mins) if row.avg_time_to_outcome_mins else 0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting win rate by stage: {e}")
            return []

    def get_missed_profit_analysis(self, days: int = 30, pair: str = None) -> Dict[str, Any]:
        """
        Get missed profit analysis with breakdowns.

        Args:
            days: Number of days to include
            pair: Optional pair filter

        Returns:
            Dictionary with missed profit analysis
        """
        pair_filter = "AND pair = :pair" if pair else ""

        query = text(f"""
        SELECT
            pair,
            rejection_stage,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as missed_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as avoided_losers,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(AVG(CASE WHEN outcome = 'HIT_TP' THEN time_to_outcome_minutes END)::numeric, 0) as avg_time_to_tp_mins
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
        {pair_filter}
        GROUP BY pair, rejection_stage
        ORDER BY missed_profit_pips DESC
        """)

        params = {'days_interval': f'{days} days'}
        if pair:
            params['pair'] = pair

        try:
            result = self.db.execute(query, params)
            rows = result.fetchall()

            breakdown = [
                {
                    'pair': row.pair,
                    'rejection_stage': row.rejection_stage,
                    'total_analyzed': row.total_analyzed,
                    'missed_winners': row.missed_winners,
                    'avoided_losers': row.avoided_losers,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'missed_profit_pips': float(row.missed_profit_pips) if row.missed_profit_pips else 0.0,
                    'avoided_loss_pips': float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0,
                    'avg_time_to_tp_mins': int(row.avg_time_to_tp_mins) if row.avg_time_to_tp_mins else 0
                }
                for row in rows
            ]

            # Calculate totals
            total_missed_profit = sum(r['missed_profit_pips'] for r in breakdown)
            total_avoided_loss = sum(r['avoided_loss_pips'] for r in breakdown)
            total_missed_winners = sum(r['missed_winners'] for r in breakdown)
            total_avoided_losers = sum(r['avoided_losers'] for r in breakdown)

            return {
                'total_missed_profit_pips': round(total_missed_profit, 1),
                'total_avoided_loss_pips': round(total_avoided_loss, 1),
                'net_impact_pips': round(total_missed_profit - total_avoided_loss, 1),
                'total_missed_winners': total_missed_winners,
                'total_avoided_losers': total_avoided_losers,
                'breakdown': breakdown,
                'days': days,
                'pair_filter': pair
            }

        except Exception as e:
            logger.error(f"Error getting missed profit analysis: {e}")
            return {'total_missed_profit_pips': 0, 'error': str(e)}

    def get_outcome_by_session(self, days: int = 30) -> List[Dict]:
        """
        Get outcome breakdown by market session.

        Args:
            days: Number of days to include

        Returns:
            List of dictionaries with session-level metrics
        """
        query = text("""
        SELECT
            market_session,
            rejection_stage,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_potential_pips
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
          AND market_session IS NOT NULL
        GROUP BY market_session, rejection_stage
        ORDER BY market_session, total_analyzed DESC
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            rows = result.fetchall()

            return [
                {
                    'market_session': row.market_session,
                    'rejection_stage': row.rejection_stage,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'net_potential_pips': float(row.net_potential_pips) if row.net_potential_pips else 0.0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting outcome by session: {e}")
            return []

    def get_outcome_by_hour(self, days: int = 30) -> List[Dict]:
        """
        Get outcome breakdown by hour of day.

        Args:
            days: Number of days to include

        Returns:
            List of dictionaries with hourly metrics
        """
        query = text("""
        SELECT
            market_hour,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
          AND market_hour IS NOT NULL
        GROUP BY market_hour
        ORDER BY market_hour
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            rows = result.fetchall()

            return [
                {
                    'market_hour': row.market_hour,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting outcome by hour: {e}")
            return []

    def get_outcome_by_pair(self, days: int = 30) -> List[Dict]:
        """
        Get outcome breakdown by currency pair (epic).

        Args:
            days: Number of days to include

        Returns:
            List of dictionaries with pair-level metrics and recommendations
        """
        query = text("""
        SELECT
            epic,
            pair,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            COUNT(CASE WHEN outcome = 'STILL_OPEN' THEN 1 END) as still_open,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_pips,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe_pips,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae_pips,
            ROUND(AVG(time_to_outcome_minutes)::numeric, 0) as avg_time_to_outcome_mins
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
        GROUP BY epic, pair
        ORDER BY total_analyzed DESC
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            rows = result.fetchall()

            pair_data = []
            for row in rows:
                win_rate = float(row.would_be_win_rate) if row.would_be_win_rate else 0.0
                missed_profit = float(row.missed_profit_pips) if row.missed_profit_pips else 0.0
                avoided_loss = float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0
                net_pips = float(row.net_pips) if row.net_pips else 0.0

                # Generate recommendation based on win rate
                if row.total_analyzed < 10:
                    recommendation = "Insufficient data for recommendation"
                    status = "NEUTRAL"
                elif win_rate > 60:
                    recommendation = f"Filters too aggressive - missing {missed_profit:.0f} pips profit. Consider relaxing filters for {row.pair}."
                    status = "TOO_AGGRESSIVE"
                elif win_rate < 40:
                    recommendation = f"Filters working well - avoided {avoided_loss:.0f} pips loss. Keep current settings for {row.pair}."
                    status = "WORKING_WELL"
                else:
                    recommendation = f"Neutral performance. Net impact: {net_pips:.0f} pips. Consider fine-tuning for {row.pair}."
                    status = "NEUTRAL"

                pair_data.append({
                    'epic': row.epic,
                    'pair': row.pair,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'still_open': row.still_open,
                    'would_be_win_rate': win_rate,
                    'missed_profit_pips': missed_profit,
                    'avoided_loss_pips': avoided_loss,
                    'net_pips': net_pips,
                    'avg_mfe_pips': float(row.avg_mfe_pips) if row.avg_mfe_pips else 0.0,
                    'avg_mae_pips': float(row.avg_mae_pips) if row.avg_mae_pips else 0.0,
                    'avg_time_to_outcome_mins': int(row.avg_time_to_outcome_mins) if row.avg_time_to_outcome_mins else 0,
                    'status': status,
                    'recommendation': recommendation
                })

            return pair_data

        except Exception as e:
            logger.error(f"Error getting outcome by pair: {e}")
            return []

    def get_pair_stage_breakdown(self, days: int = 30, pair: str = None) -> List[Dict]:
        """
        Get detailed breakdown by pair and stage for specific pair analysis.

        Args:
            days: Number of days to include
            pair: Optional pair filter (e.g., 'EURUSD')

        Returns:
            List of dictionaries with pair+stage level metrics
        """
        pair_filter = "AND pair = :pair" if pair else ""

        query = text(f"""
        SELECT
            epic,
            pair,
            rejection_stage,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_pips
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
        {pair_filter}
        GROUP BY epic, pair, rejection_stage
        ORDER BY pair, total_analyzed DESC
        """)

        params = {'days_interval': f'{days} days'}
        if pair:
            params['pair'] = pair

        try:
            result = self.db.execute(query, params)
            rows = result.fetchall()

            return [
                {
                    'epic': row.epic,
                    'pair': row.pair,
                    'rejection_stage': row.rejection_stage,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'missed_profit_pips': float(row.missed_profit_pips) if row.missed_profit_pips else 0.0,
                    'avoided_loss_pips': float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0,
                    'net_pips': float(row.net_pips) if row.net_pips else 0.0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting pair stage breakdown: {e}")
            return []

    def get_mfe_mae_analysis(self, days: int = 30, stage: str = None) -> Dict[str, Any]:
        """
        Get MFE/MAE statistics for strategy tuning.

        Args:
            days: Number of days to include
            stage: Optional rejection stage filter

        Returns:
            Dictionary with MFE/MAE analysis
        """
        stage_filter = "AND rejection_stage = :stage" if stage else ""

        query = text(f"""
        SELECT
            outcome,
            COUNT(*) as count,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe,
            ROUND(MAX(max_favorable_excursion_pips)::numeric, 2) as max_mfe,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae,
            ROUND(MAX(max_adverse_excursion_pips)::numeric, 2) as max_mae,
            ROUND(AVG(time_to_mfe_minutes)::numeric, 0) as avg_time_to_mfe,
            ROUND(AVG(time_to_mae_minutes)::numeric, 0) as avg_time_to_mae
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
          AND outcome IN ('HIT_TP', 'HIT_SL')
        {stage_filter}
        GROUP BY outcome
        """)

        params = {'days_interval': f'{days} days'}
        if stage:
            params['stage'] = stage

        try:
            result = self.db.execute(query, params)
            rows = result.fetchall()

            data = {}
            for row in rows:
                data[row.outcome] = {
                    'count': row.count,
                    'avg_mfe': float(row.avg_mfe) if row.avg_mfe else 0.0,
                    'max_mfe': float(row.max_mfe) if row.max_mfe else 0.0,
                    'avg_mae': float(row.avg_mae) if row.avg_mae else 0.0,
                    'max_mae': float(row.max_mae) if row.max_mae else 0.0,
                    'avg_time_to_mfe': int(row.avg_time_to_mfe) if row.avg_time_to_mfe else 0,
                    'avg_time_to_mae': int(row.avg_time_to_mae) if row.avg_time_to_mae else 0
                }

            return {
                'winners': data.get('HIT_TP', {}),
                'losers': data.get('HIT_SL', {}),
                'days': days,
                'stage_filter': stage
            }

        except Exception as e:
            logger.error(f"Error getting MFE/MAE analysis: {e}")
            return {'error': str(e)}

    def get_parameter_suggestions(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate AI-ready parameter suggestions based on outcome patterns.

        Analyzes:
        - Which stages reject too many winners (lower threshold)
        - Which stages let through too many losers (raise threshold)
        - Session/hour patterns that could improve timing

        Args:
            days: Number of days to include

        Returns:
            Dictionary with actionable suggestions
        """
        suggestions = {
            'stage_adjustments': [],
            'session_patterns': [],
            'pair_insights': [],
            'overall_assessment': ''
        }

        # Get stage data
        stage_data = self.get_win_rate_by_stage(days)

        if not stage_data:
            suggestions['overall_assessment'] = 'No outcome data available yet. Run the analyzer to populate data.'
            return suggestions

        # Analyze each stage
        for stage in stage_data:
            win_rate = stage.get('would_be_win_rate', 0)
            total = stage.get('total_analyzed', 0)
            missed_profit = stage.get('missed_profit_pips', 0)
            avoided_loss = stage.get('avoided_loss_pips', 0)
            stage_name = stage.get('rejection_stage', 'UNKNOWN')

            if total < 10:
                continue  # Skip stages with too few samples

            if win_rate > 60:
                # Stage is too aggressive - rejecting profitable signals
                suggestion = {
                    'stage': stage_name,
                    'issue': 'TOO_AGGRESSIVE',
                    'win_rate': win_rate,
                    'total_rejections': total,
                    'missed_profit_pips': missed_profit,
                    'recommendation': f'{stage_name} rejects {win_rate:.0f}% would-be winners. '
                                     f'Consider relaxing this filter to capture {missed_profit:.0f} more pips.'
                }
                suggestions['stage_adjustments'].append(suggestion)

            elif win_rate < 40:
                # Stage is working well - filtering losers
                suggestion = {
                    'stage': stage_name,
                    'issue': 'WORKING_WELL',
                    'win_rate': win_rate,
                    'total_rejections': total,
                    'avoided_loss_pips': avoided_loss,
                    'recommendation': f'{stage_name} correctly filters {100 - win_rate:.0f}% losers, '
                                     f'saving {avoided_loss:.0f} pips. Keep current settings.'
                }
                suggestions['stage_adjustments'].append(suggestion)

            else:
                # Neutral performance
                suggestion = {
                    'stage': stage_name,
                    'issue': 'NEUTRAL',
                    'win_rate': win_rate,
                    'total_rejections': total,
                    'recommendation': f'{stage_name} has neutral {win_rate:.0f}% win rate. '
                                     f'Net impact: {missed_profit - avoided_loss:.0f} pips. Consider fine-tuning.'
                }
                suggestions['stage_adjustments'].append(suggestion)

        # Get session patterns
        session_data = self.get_outcome_by_session(days)
        session_summary = {}

        for item in session_data:
            session = item.get('market_session')
            if session not in session_summary:
                session_summary[session] = {'winners': 0, 'losers': 0}
            session_summary[session]['winners'] += item.get('would_be_winners', 0)
            session_summary[session]['losers'] += item.get('would_be_losers', 0)

        for session, data in session_summary.items():
            total = data['winners'] + data['losers']
            if total >= 10:
                win_rate = (data['winners'] / total) * 100 if total > 0 else 0
                suggestions['session_patterns'].append({
                    'session': session,
                    'win_rate': round(win_rate, 1),
                    'total': total,
                    'insight': 'High potential' if win_rate > 55 else 'Standard' if win_rate > 45 else 'Review filters'
                })

        # Get pair-specific insights
        pair_data = self.get_outcome_by_pair(days)
        for pair_info in pair_data:
            if pair_info.get('total_analyzed', 0) >= 10:
                suggestions['pair_insights'].append({
                    'epic': pair_info['epic'],
                    'pair': pair_info['pair'],
                    'win_rate': pair_info['would_be_win_rate'],
                    'total_analyzed': pair_info['total_analyzed'],
                    'missed_profit_pips': pair_info['missed_profit_pips'],
                    'avoided_loss_pips': pair_info['avoided_loss_pips'],
                    'net_pips': pair_info['net_pips'],
                    'status': pair_info['status'],
                    'recommendation': pair_info['recommendation']
                })

        # Overall assessment
        total_analyzed = sum(s.get('total_analyzed', 0) for s in stage_data)
        total_missed = sum(s.get('missed_profit_pips', 0) for s in stage_data)
        total_avoided = sum(s.get('avoided_loss_pips', 0) for s in stage_data)
        net_impact = total_missed - total_avoided

        if net_impact > 0:
            suggestions['overall_assessment'] = (
                f'Over {days} days: Filters are net-negative by {net_impact:.0f} pips. '
                f'Missed {total_missed:.0f} pips of profit while avoiding {total_avoided:.0f} pips of loss. '
                f'Consider relaxing aggressive filters.'
            )
        else:
            suggestions['overall_assessment'] = (
                f'Over {days} days: Filters are net-positive by {abs(net_impact):.0f} pips. '
                f'Avoided {total_avoided:.0f} pips of loss while missing {total_missed:.0f} pips of profit. '
                f'Current filter settings are protective.'
            )

        return suggestions

    def get_outcome_by_pair_and_direction(self, days: int = 30) -> List[Dict]:
        """
        Get outcome breakdown by currency pair AND direction (BULL/BEAR).

        This is critical for direction-aware parameter tuning - allows different
        filter thresholds for BULL vs BEAR trades per pair.

        Args:
            days: Number of days to include

        Returns:
            List of dictionaries with pair+direction level metrics
        """
        query = text("""
        SELECT
            epic,
            pair,
            attempted_direction,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            COUNT(CASE WHEN outcome = 'STILL_OPEN' THEN 1 END) as still_open,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_pips,
            ROUND(AVG(max_favorable_excursion_pips)::numeric, 2) as avg_mfe_pips,
            ROUND(AVG(max_adverse_excursion_pips)::numeric, 2) as avg_mae_pips
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
          AND attempted_direction IS NOT NULL
        GROUP BY epic, pair, attempted_direction
        ORDER BY pair, attempted_direction
        """)

        try:
            result = self.db.execute(query, {'days_interval': f'{days} days'})
            rows = result.fetchall()

            pair_direction_data = []
            for row in rows:
                win_rate = float(row.would_be_win_rate) if row.would_be_win_rate else 0.0
                missed_profit = float(row.missed_profit_pips) if row.missed_profit_pips else 0.0
                avoided_loss = float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0
                net_pips = float(row.net_pips) if row.net_pips else 0.0

                # Generate direction-aware recommendation
                if row.total_analyzed < 5:
                    recommendation = "Insufficient data for recommendation"
                    status = "NEUTRAL"
                elif win_rate > 60:
                    recommendation = f"Filters too aggressive for {row.pair} {row.attempted_direction} - missing {missed_profit:.0f} pips. Relax {row.attempted_direction.lower()} direction filters."
                    status = "TOO_AGGRESSIVE"
                elif win_rate < 40:
                    recommendation = f"Filters working well for {row.pair} {row.attempted_direction} - avoided {avoided_loss:.0f} pips loss. Keep strict filters."
                    status = "WORKING_WELL"
                else:
                    recommendation = f"Neutral performance for {row.pair} {row.attempted_direction}. Net impact: {net_pips:.0f} pips."
                    status = "NEUTRAL"

                pair_direction_data.append({
                    'epic': row.epic,
                    'pair': row.pair,
                    'direction': row.attempted_direction,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'still_open': row.still_open,
                    'would_be_win_rate': win_rate,
                    'missed_profit_pips': missed_profit,
                    'avoided_loss_pips': avoided_loss,
                    'net_pips': net_pips,
                    'avg_mfe_pips': float(row.avg_mfe_pips) if row.avg_mfe_pips else 0.0,
                    'avg_mae_pips': float(row.avg_mae_pips) if row.avg_mae_pips else 0.0,
                    'status': status,
                    'recommendation': recommendation
                })

            return pair_direction_data

        except Exception as e:
            logger.error(f"Error getting outcome by pair and direction: {e}")
            return []

    def get_pair_direction_stage_breakdown(self, days: int = 30, pair: str = None, direction: str = None) -> List[Dict]:
        """
        Get detailed breakdown by pair, direction, and rejection stage.

        This allows analyzing which specific stages cause issues for
        BULL vs BEAR trades on each pair.

        Args:
            days: Number of days to include
            pair: Optional pair filter (e.g., 'EURUSD')
            direction: Optional direction filter ('BULL' or 'BEAR')

        Returns:
            List of dictionaries with pair+direction+stage level metrics
        """
        filters = []
        params = {'days_interval': f'{days} days'}

        if pair:
            filters.append("pair = :pair")
            params['pair'] = pair
        if direction:
            filters.append("attempted_direction = :direction")
            params['direction'] = direction

        where_clause = f"AND {' AND '.join(filters)}" if filters else ""

        query = text(f"""
        SELECT
            epic,
            pair,
            attempted_direction,
            rejection_stage,
            rejection_reason,
            COUNT(*) as total_analyzed,
            COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END) as would_be_winners,
            COUNT(CASE WHEN outcome = 'HIT_SL' THEN 1 END) as would_be_losers,
            ROUND(
                COUNT(CASE WHEN outcome = 'HIT_TP' THEN 1 END)::numeric /
                NULLIF(COUNT(CASE WHEN outcome IN ('HIT_TP', 'HIT_SL') THEN 1 END), 0) * 100,
                1
            ) as would_be_win_rate,
            ROUND(SUM(CASE WHEN outcome = 'HIT_TP' THEN potential_profit_pips ELSE 0 END)::numeric, 1) as missed_profit_pips,
            ROUND(SUM(CASE WHEN outcome = 'HIT_SL' THEN ABS(potential_profit_pips) ELSE 0 END)::numeric, 1) as avoided_loss_pips,
            ROUND(SUM(potential_profit_pips)::numeric, 1) as net_pips
        FROM smc_rejection_outcomes
        WHERE analysis_timestamp >= NOW() - INTERVAL :days_interval
          AND attempted_direction IS NOT NULL
        {where_clause}
        GROUP BY epic, pair, attempted_direction, rejection_stage, rejection_reason
        ORDER BY pair, attempted_direction, total_analyzed DESC
        """)

        try:
            result = self.db.execute(query, params)
            rows = result.fetchall()

            return [
                {
                    'epic': row.epic,
                    'pair': row.pair,
                    'direction': row.attempted_direction,
                    'rejection_stage': row.rejection_stage,
                    'rejection_reason': row.rejection_reason,
                    'total_analyzed': row.total_analyzed,
                    'would_be_winners': row.would_be_winners,
                    'would_be_losers': row.would_be_losers,
                    'would_be_win_rate': float(row.would_be_win_rate) if row.would_be_win_rate else 0.0,
                    'missed_profit_pips': float(row.missed_profit_pips) if row.missed_profit_pips else 0.0,
                    'avoided_loss_pips': float(row.avoided_loss_pips) if row.avoided_loss_pips else 0.0,
                    'net_pips': float(row.net_pips) if row.net_pips else 0.0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting pair direction stage breakdown: {e}")
            return []

    def get_direction_aware_suggestions(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate direction-aware parameter suggestions.

        Analyzes BULL vs BEAR performance separately for each pair
        to recommend direction-specific filter adjustments.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with direction-aware suggestions
        """
        suggestions = {
            'direction_recommendations': [],
            'pairs_needing_direction_config': [],
            'overall_direction_summary': {}
        }

        # Get pair+direction data
        pair_direction_data = self.get_outcome_by_pair_and_direction(days)

        if not pair_direction_data:
            return suggestions

        # Group by pair to compare BULL vs BEAR
        pair_data = {}
        for item in pair_direction_data:
            epic = item['epic']
            if epic not in pair_data:
                pair_data[epic] = {'pair': item['pair'], 'BULL': None, 'BEAR': None}
            pair_data[epic][item['direction']] = item

        # Analyze each pair's directional performance
        for epic, data in pair_data.items():
            pair_name = data['pair']
            bull = data.get('BULL')
            bear = data.get('BEAR')

            # Check if there's significant difference between directions
            if bull and bear:
                bull_wr = bull.get('would_be_win_rate', 50)
                bear_wr = bear.get('would_be_win_rate', 50)
                bull_total = bull.get('total_analyzed', 0)
                bear_total = bear.get('total_analyzed', 0)

                # Only analyze if we have enough data for both
                if bull_total >= 5 and bear_total >= 5:
                    wr_diff = abs(bull_wr - bear_wr)

                    if wr_diff >= 20:  # Significant difference (20+ percentage points)
                        # This pair needs direction-aware config
                        suggestions['pairs_needing_direction_config'].append({
                            'epic': epic,
                            'pair': pair_name,
                            'bull_win_rate': bull_wr,
                            'bear_win_rate': bear_wr,
                            'bull_total': bull_total,
                            'bear_total': bear_total,
                            'bull_missed_pips': bull.get('missed_profit_pips', 0),
                            'bear_missed_pips': bear.get('missed_profit_pips', 0),
                            'difference': wr_diff,
                            'recommended_action': self._get_direction_recommendation(bull_wr, bear_wr, pair_name)
                        })

            # Generate individual direction recommendations
            for direction_data in [bull, bear]:
                if direction_data and direction_data.get('total_analyzed', 0) >= 5:
                    win_rate = direction_data.get('would_be_win_rate', 50)
                    direction = direction_data.get('direction')

                    if win_rate > 60:
                        suggestions['direction_recommendations'].append({
                            'epic': epic,
                            'pair': pair_name,
                            'direction': direction,
                            'action': 'relax',
                            'win_rate': win_rate,
                            'missed_pips': direction_data.get('missed_profit_pips', 0),
                            'total_analyzed': direction_data.get('total_analyzed', 0),
                            'recommendation': f"Relax filters for {pair_name} {direction} trades (WR: {win_rate:.0f}%)",
                            'suggested_params': self._suggest_direction_params(direction, 'relax')
                        })
                    elif win_rate < 40:
                        suggestions['direction_recommendations'].append({
                            'epic': epic,
                            'pair': pair_name,
                            'direction': direction,
                            'action': 'keep_strict',
                            'win_rate': win_rate,
                            'avoided_pips': direction_data.get('avoided_loss_pips', 0),
                            'total_analyzed': direction_data.get('total_analyzed', 0),
                            'recommendation': f"Keep strict filters for {pair_name} {direction} trades (WR: {win_rate:.0f}%)",
                            'suggested_params': None
                        })

        # Overall direction summary
        all_bull = [d for d in pair_direction_data if d['direction'] == 'BULL']
        all_bear = [d for d in pair_direction_data if d['direction'] == 'BEAR']

        if all_bull:
            total_bull = sum(d['total_analyzed'] for d in all_bull)
            avg_bull_wr = sum(d['would_be_win_rate'] * d['total_analyzed'] for d in all_bull) / total_bull if total_bull > 0 else 0
            suggestions['overall_direction_summary']['BULL'] = {
                'total_analyzed': total_bull,
                'avg_win_rate': round(avg_bull_wr, 1),
                'total_missed_pips': sum(d['missed_profit_pips'] for d in all_bull),
                'total_avoided_pips': sum(d['avoided_loss_pips'] for d in all_bull)
            }

        if all_bear:
            total_bear = sum(d['total_analyzed'] for d in all_bear)
            avg_bear_wr = sum(d['would_be_win_rate'] * d['total_analyzed'] for d in all_bear) / total_bear if total_bear > 0 else 0
            suggestions['overall_direction_summary']['BEAR'] = {
                'total_analyzed': total_bear,
                'avg_win_rate': round(avg_bear_wr, 1),
                'total_missed_pips': sum(d['missed_profit_pips'] for d in all_bear),
                'total_avoided_pips': sum(d['avoided_loss_pips'] for d in all_bear)
            }

        return suggestions

    def _get_direction_recommendation(self, bull_wr: float, bear_wr: float, pair: str) -> str:
        """Generate recommendation based on directional win rate difference."""
        if bull_wr > bear_wr + 20:
            return f"Enable direction overrides for {pair}: Relax BULL filters, keep BEAR strict"
        elif bear_wr > bull_wr + 20:
            return f"Enable direction overrides for {pair}: Relax BEAR filters, keep BULL strict"
        else:
            return f"Direction performance similar for {pair}"

    def _suggest_direction_params(self, direction: str, action: str) -> Dict[str, Any]:
        """Suggest specific parameter values for direction-aware config."""
        dir_suffix = direction.lower()

        if action == 'relax':
            return {
                f'fib_pullback_min_{dir_suffix}': 0.10,  # Lower from 0.236
                f'fib_pullback_max_{dir_suffix}': 0.85,  # Higher from 0.70
                f'momentum_min_depth_{dir_suffix}': -0.80,  # More negative from -0.45
                f'min_volume_ratio_{dir_suffix}': 0.20,  # Lower from 0.50
            }
        return None

    def get_recent_outcomes(self, limit: int = 100, filters: Dict = None) -> List[Dict]:
        """
        Get recent individual outcome records with optional filters.

        Args:
            limit: Maximum number of records
            filters: Optional filter dictionary

        Returns:
            List of outcome records
        """
        where_clauses = ["1=1"]
        params = {'limit': limit}

        if filters:
            if filters.get('stage'):
                where_clauses.append("rejection_stage = :stage")
                params['stage'] = filters['stage']
            if filters.get('outcome'):
                where_clauses.append("outcome = :outcome")
                params['outcome'] = filters['outcome']
            if filters.get('pair'):
                where_clauses.append("pair = :pair")
                params['pair'] = filters['pair']

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
        SELECT
            id,
            rejection_id,
            epic,
            pair,
            rejection_timestamp,
            rejection_stage,
            attempted_direction,
            market_session,
            entry_price,
            stop_loss_price,
            take_profit_price,
            outcome,
            outcome_price,
            outcome_timestamp,
            time_to_outcome_minutes,
            max_favorable_excursion_pips,
            max_adverse_excursion_pips,
            potential_profit_pips,
            analysis_timestamp
        FROM smc_rejection_outcomes
        WHERE {where_sql}
        ORDER BY rejection_timestamp DESC
        LIMIT :limit
        """)

        try:
            result = self.db.execute(query, params)
            rows = result.fetchall()

            return [
                {
                    'id': row.id,
                    'rejection_id': row.rejection_id,
                    'epic': row.epic,
                    'pair': row.pair,
                    'rejection_timestamp': row.rejection_timestamp.isoformat() if row.rejection_timestamp else None,
                    'rejection_stage': row.rejection_stage,
                    'attempted_direction': row.attempted_direction,
                    'market_session': row.market_session,
                    'entry_price': float(row.entry_price) if row.entry_price else None,
                    'stop_loss_price': float(row.stop_loss_price) if row.stop_loss_price else None,
                    'take_profit_price': float(row.take_profit_price) if row.take_profit_price else None,
                    'outcome': row.outcome,
                    'outcome_price': float(row.outcome_price) if row.outcome_price else None,
                    'outcome_timestamp': row.outcome_timestamp.isoformat() if row.outcome_timestamp else None,
                    'time_to_outcome_minutes': row.time_to_outcome_minutes,
                    'max_favorable_excursion_pips': float(row.max_favorable_excursion_pips) if row.max_favorable_excursion_pips else 0,
                    'max_adverse_excursion_pips': float(row.max_adverse_excursion_pips) if row.max_adverse_excursion_pips else 0,
                    'potential_profit_pips': float(row.potential_profit_pips) if row.potential_profit_pips else 0,
                    'analysis_timestamp': row.analysis_timestamp.isoformat() if row.analysis_timestamp else None
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting recent outcomes: {e}")
            return []
