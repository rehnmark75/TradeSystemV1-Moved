# optimization/generators/report_generator.py
"""
Report generator for unified parameter optimizer.
Formats console output and summary reports.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from ..analyzers.correlation_analyzer import OptimizationRecommendation
from ..analyzers.direction_analyzer import DirectionPerformance, DirectionRecommendation
from ..analyzers.regime_analyzer import RegimePerformance, RegimeFilterRecommendation


class ReportGenerator:
    """Generates formatted reports for optimization results"""

    # Box drawing characters
    BOX_H = '─'
    BOX_V = '│'
    BOX_TL = '┌'
    BOX_TR = '┐'
    BOX_BL = '└'
    BOX_BR = '┘'
    BOX_ML = '├'
    BOX_MR = '┤'
    BOX_MT = '┬'
    BOX_MB = '┴'
    BOX_X = '┼'

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_header(self, days: int, epics_count: int, trades_count: int, rejections_count: int) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return f"""
{'═' * 80}
                    UNIFIED PARAMETER OPTIMIZATION REPORT
                    Period: {days} days | Generated: {timestamp}
{'═' * 80}

SUMMARY:
  Epics Analyzed: {epics_count}
  Total Trades: {trades_count}
  Rejections Analyzed: {rejections_count}
"""

    def generate_epic_section(
        self,
        epic: str,
        full_epic: str,
        trade_summary: Dict[str, Any],
        direction_perf: Dict[str, DirectionPerformance] = None,
        regime_perf: Dict[str, RegimePerformance] = None,
        recommendations: List[OptimizationRecommendation] = None,
        direction_recs: List[DirectionRecommendation] = None,
        regime_rec: RegimeFilterRecommendation = None,
        sql: str = None
    ) -> str:
        """Generate section for a single epic"""
        parts = []

        # Epic header
        parts.append(f"""
{'═' * 80}
{epic} ({full_epic})
{'═' * 80}""")

        # Performance summary
        if trade_summary:
            wr = trade_summary.get('win_rate', 0)
            trades = trade_summary.get('total_trades', 0)
            pf = trade_summary.get('profit_factor', 0)
            avg_pips = trade_summary.get('avg_pips', 0)
            parts.append(f"PERFORMANCE: {wr:.1%} WR ({trades} trades) | PF: {pf:.2f} | Avg: {avg_pips:+.1f} pips")

        # Direction analysis
        if direction_perf:
            parts.append("\nDIRECTION ANALYSIS:")
            for direction, perf in sorted(direction_perf.items()):
                status = self._get_direction_status(perf.win_rate, direction_perf)
                parts.append(f"  {direction}: {perf.win_rate:.1%} WR ({perf.total_trades} trades) {status}")

            if direction_recs:
                for rec in direction_recs:
                    if rec.epic == epic or rec.epic in full_epic:
                        parts.append(f"  -> {rec.reason}")

        # Regime analysis
        if regime_perf:
            parts.append("\nREGIME ANALYSIS:")
            sorted_regimes = sorted(regime_perf.items(), key=lambda x: x[1].win_rate, reverse=True)
            for regime, perf in sorted_regimes:
                status = self._get_regime_status(perf)
                parts.append(f"  {regime:12}: {perf.win_rate:.1%} WR ({perf.total_trades} trades) {status}")

            if regime_rec:
                parts.append(f"  -> Recommend blocking: {', '.join(regime_rec.block_regimes)}")

        # Parameter recommendations table
        if recommendations:
            epic_recs = [r for r in recommendations if r.epic == epic or r.epic in full_epic]
            if epic_recs:
                parts.append("\nPARAMETER RECOMMENDATIONS:")
                parts.append(self._build_recommendation_table(epic_recs))

        # SQL
        if sql and sql.strip():
            parts.append("\nSQL (copy to apply):")
            parts.append('─' * 80)
            parts.append(sql)

        return "\n".join(parts)

    def _get_direction_status(
        self,
        win_rate: float,
        all_perf: Dict[str, DirectionPerformance]
    ) -> str:
        """Get status emoji/text for direction"""
        other_wr = [p.win_rate for d, p in all_perf.items()]
        avg_wr = sum(other_wr) / len(other_wr) if other_wr else 0.5

        if win_rate >= avg_wr + 0.10:
            return "-> Relax filters"
        elif win_rate <= avg_wr - 0.10:
            return "-> Tighten filters"
        return ""

    def _get_regime_status(self, perf: RegimePerformance) -> str:
        """Get status emoji/text for regime"""
        if perf.win_rate >= 0.55:
            return "OK"
        elif perf.win_rate >= 0.45:
            return "Caution"
        else:
            return "BLOCK"

    def _build_recommendation_table(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Build ASCII table for recommendations"""
        # Column widths
        col_widths = [23, 9, 13, 12, 14]
        headers = ['Parameter', 'Current', 'Recommended', 'Confidence', 'Est. Impact']

        # Build table
        lines = []

        # Header row
        header_line = self.BOX_TL
        for i, width in enumerate(col_widths):
            header_line += self.BOX_H * width
            header_line += self.BOX_MT if i < len(col_widths) - 1 else self.BOX_TR
        lines.append(header_line)

        # Header text
        header_text = self.BOX_V
        for header, width in zip(headers, col_widths):
            header_text += f" {header:<{width-1}}{self.BOX_V}"
        lines.append(header_text)

        # Separator
        sep_line = self.BOX_ML
        for i, width in enumerate(col_widths):
            sep_line += self.BOX_H * width
            sep_line += self.BOX_X if i < len(col_widths) - 1 else self.BOX_MR
        lines.append(sep_line)

        # Data rows
        for rec in recommendations:
            impact_str = f"+{rec.expected_impact_pips:.0f} pips/mo" if rec.expected_impact_pips > 0 else "N/A"

            data_line = self.BOX_V
            data_line += f" {rec.parameter[:22]:<22}{self.BOX_V}"
            data_line += f" {str(rec.current_value)[:8]:>8} {self.BOX_V}"
            data_line += f" {str(rec.recommended_value)[:12]:>12}{self.BOX_V}"
            data_line += f" {rec.confidence:>10.0%} {self.BOX_V}"
            data_line += f" {impact_str:>13}{self.BOX_V}"
            lines.append(data_line)

        # Bottom border
        bottom_line = self.BOX_BL
        for i, width in enumerate(col_widths):
            bottom_line += self.BOX_H * width
            bottom_line += self.BOX_MB if i < len(col_widths) - 1 else self.BOX_BR
        lines.append(bottom_line)

        return "\n".join(lines)

    def generate_summary(
        self,
        all_recommendations: List[OptimizationRecommendation],
        direction_recs: List[DirectionRecommendation],
        regime_recs: List[RegimeFilterRecommendation],
        min_confidence: float
    ) -> str:
        """Generate final summary"""
        high_conf_recs = [r for r in all_recommendations if r.confidence >= 0.80]
        high_conf_dir = [r for r in direction_recs if r.confidence >= 0.80] if direction_recs else []
        high_conf_regime = [r for r in regime_recs if r.confidence >= 0.80] if regime_recs else []

        total_impact = sum(r.expected_impact_pips for r in all_recommendations if r.expected_impact_pips > 0)

        return f"""
{'═' * 80}
SUMMARY
{'═' * 80}

Total Recommendations: {len(all_recommendations)}
  - High Confidence (>80%): {len(high_conf_recs)}
  - Direction-specific: {len(direction_recs) if direction_recs else 0}
  - Regime Filters: {len(regime_recs) if regime_recs else 0}

Estimated Total Impact: +{total_impact:.0f} pips/month

NEXT STEPS:
  1. Review recommendations above
  2. Copy SQL statements to apply changes
  3. Run: docker exec postgres psql -U postgres -d strategy_config -f <file>
  4. Restart scanner: docker restart task-worker

{'═' * 80}
"""

    def generate_full_report(
        self,
        days: int,
        trade_summaries: Dict[str, Dict[str, Any]],
        direction_perfs: Dict[str, Dict[str, DirectionPerformance]],
        regime_perfs: Dict[str, Dict[str, RegimePerformance]],
        recommendations: List[OptimizationRecommendation],
        direction_recs: List[DirectionRecommendation],
        regime_recs: List[RegimeFilterRecommendation],
        sql_by_epic: Dict[str, str],
        min_confidence: float = 0.70,
        rejection_stats: Dict[str, Any] = None
    ) -> str:
        """Generate complete report"""
        parts = []

        # Calculate totals - use rejection_stats if trade_summaries is empty
        if rejection_stats:
            epics_count = rejection_stats.get('epics_count', 0)
            rejections_count = rejection_stats.get('total_rejections', 0)
        else:
            epics_count = len(trade_summaries)
            rejections_count = 0

        # If we have trade summaries, use those for epic count
        if trade_summaries:
            epics_count = max(epics_count, len(trade_summaries))

        trades_count = sum(s.get('total_trades', 0) for s in trade_summaries.values())

        # Header
        parts.append(self.generate_header(days, epics_count, trades_count, rejections_count))

        # Epic sections
        epic_mapping = {
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

        # Build list of epics to report on (from trades, rejections, or recommendations)
        epics_to_report = set(trade_summaries.keys())

        # Add epics from rejection_stats if available
        if rejection_stats and 'epics' in rejection_stats:
            for full_epic in rejection_stats['epics']:
                # Extract short name from full epic
                for short, full in epic_mapping.items():
                    if full == full_epic:
                        epics_to_report.add(short)
                        break
                else:
                    epics_to_report.add(full_epic)

        # Add epics from recommendations
        if recommendations:
            for rec in recommendations:
                for short, full in epic_mapping.items():
                    if full == rec.epic:
                        epics_to_report.add(short)
                        break
                else:
                    epics_to_report.add(rec.epic)

        for epic in sorted(epics_to_report):
            full_epic = epic_mapping.get(epic, epic)

            # Get regime recommendation for this epic
            epic_regime_rec = None
            if regime_recs:
                for rec in regime_recs:
                    if rec.epic == epic or rec.epic == full_epic:
                        epic_regime_rec = rec
                        break

            parts.append(self.generate_epic_section(
                epic=epic,
                full_epic=full_epic,
                trade_summary=trade_summaries.get(epic, {}),
                direction_perf=direction_perfs.get(epic),
                regime_perf=regime_perfs.get(epic),
                recommendations=recommendations,
                direction_recs=direction_recs,
                regime_rec=epic_regime_rec,
                sql=sql_by_epic.get(full_epic) or sql_by_epic.get(epic)
            ))

        # Summary
        parts.append(self.generate_summary(
            recommendations, direction_recs, regime_recs, min_confidence
        ))

        return "\n".join(parts)

    def generate_json_report(
        self,
        trade_summaries: Dict[str, Dict[str, Any]],
        recommendations: List[OptimizationRecommendation],
        direction_recs: List[DirectionRecommendation],
        regime_recs: List[RegimeFilterRecommendation]
    ) -> Dict[str, Any]:
        """Generate JSON-serializable report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'trade_summaries': trade_summaries,
            'recommendations': [
                {
                    'epic': r.epic,
                    'parameter': r.parameter,
                    'current_value': r.current_value,
                    'recommended_value': r.recommended_value,
                    'confidence': r.confidence,
                    'expected_impact_pips': r.expected_impact_pips,
                    'sample_size': r.sample_size,
                    'reason': r.reason
                }
                for r in recommendations
            ],
            'direction_recommendations': [
                {
                    'epic': r.epic,
                    'direction': r.direction,
                    'parameter': r.parameter,
                    'current_value': r.current_value,
                    'recommended_value': r.recommended_value,
                    'confidence': r.confidence,
                    'reason': r.reason
                }
                for r in (direction_recs or [])
            ],
            'regime_filter_recommendations': [
                {
                    'epic': r.epic,
                    'block_regimes': r.block_regimes,
                    'confidence': r.confidence,
                    'estimated_impact_pips': r.estimated_impact_pips,
                    'reason': r.reason
                }
                for r in (regime_recs or [])
            ]
        }
