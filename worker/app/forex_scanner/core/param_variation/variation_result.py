"""
Variation Result and Result Ranker

Data classes for storing variation results and ranking/formatting utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VariationResult:
    """Result of a single parameter variation backtest"""

    variation_index: int
    params: Dict[str, Any]

    # Core metrics
    signal_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    profit_factor: float = 0.0
    composite_score: float = 0.0

    # Additional metrics
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    max_drawdown_pips: float = 0.0
    expectancy: float = 0.0

    # Execution info
    execution_id: Optional[int] = None
    duration_seconds: float = 0.0
    status: str = 'pending'  # pending, running, completed, failed
    error: Optional[str] = None

    # Assigned after ranking
    rank: int = 0

    def calculate_derived_metrics(self):
        """Calculate derived metrics from base metrics"""
        # Win rate
        if self.signal_count > 0:
            self.win_rate = (self.win_count / self.signal_count) * 100

        # Profit factor
        total_wins = self.avg_win_pips * self.win_count if self.avg_win_pips else 0
        total_losses = abs(self.avg_loss_pips) * self.loss_count if self.avg_loss_pips else 0
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            self.profit_factor = float('inf')

        # Expectancy
        if self.signal_count > 0:
            win_prob = self.win_count / self.signal_count
            loss_prob = self.loss_count / self.signal_count
            self.expectancy = (win_prob * self.avg_win_pips) - (loss_prob * abs(self.avg_loss_pips))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'rank': self.rank,
            'variation_index': self.variation_index,
            'params': self.params,
            'signal_count': self.signal_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'total_pips': self.total_pips,
            'profit_factor': self.profit_factor,
            'composite_score': self.composite_score,
            'avg_win_pips': self.avg_win_pips,
            'avg_loss_pips': self.avg_loss_pips,
            'max_drawdown_pips': self.max_drawdown_pips,
            'expectancy': self.expectancy,
            'execution_id': self.execution_id,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'error': self.error,
        }


class ResultRanker:
    """Rank and format variation results"""

    METRICS = ['win_rate', 'total_pips', 'profit_factor', 'composite_score', 'expectancy']

    # Weights for composite score calculation
    DEFAULT_WEIGHTS = {
        'win_rate': 0.3,
        'total_pips': 0.25,
        'profit_factor': 0.25,
        'signal_count': 0.1,
        'expectancy': 0.1,
    }

    def __init__(self, score_weights: Optional[Dict[str, float]] = None):
        self.weights = score_weights or self.DEFAULT_WEIGHTS

    def calculate_composite_score(self, result: VariationResult) -> float:
        """
        Calculate composite score from multiple metrics.

        Normalizes metrics and combines them using weights.
        """
        score = 0.0

        # Win rate contribution (0-100 -> 0-1)
        if 'win_rate' in self.weights:
            win_rate_norm = min(result.win_rate / 100, 1.0)
            score += self.weights['win_rate'] * win_rate_norm

        # Profit factor contribution (cap at 5 for normalization)
        if 'profit_factor' in self.weights:
            pf_norm = min(result.profit_factor / 5, 1.0) if result.profit_factor != float('inf') else 1.0
            score += self.weights['profit_factor'] * pf_norm

        # Total pips contribution (normalize based on expected range)
        if 'total_pips' in self.weights:
            # Normalize: -50 to +100 pips mapped to 0-1
            pips_norm = max(0, min((result.total_pips + 50) / 150, 1.0))
            score += self.weights['total_pips'] * pips_norm

        # Signal count contribution (more signals = more confidence)
        if 'signal_count' in self.weights:
            # Normalize: 0-20 signals mapped to 0-1
            signals_norm = min(result.signal_count / 20, 1.0)
            score += self.weights['signal_count'] * signals_norm

        # Expectancy contribution
        if 'expectancy' in self.weights:
            # Normalize: -10 to +10 pips expected per trade
            exp_norm = max(0, min((result.expectancy + 10) / 20, 1.0))
            score += self.weights['expectancy'] * exp_norm

        return round(score, 4)

    def rank(
        self,
        results: List[VariationResult],
        by: str = 'composite_score',
        ascending: bool = False
    ) -> List[VariationResult]:
        """
        Sort results by specified metric and assign ranks.

        Args:
            results: List of VariationResult objects
            by: Metric to sort by
            ascending: Sort order (False = highest first)

        Returns:
            Sorted list with rank assigned
        """
        if not results:
            return []

        if by not in self.METRICS:
            logger.warning(f"Unknown metric '{by}', using composite_score")
            by = 'composite_score'

        # Calculate composite scores if needed
        if by == 'composite_score':
            for result in results:
                result.composite_score = self.calculate_composite_score(result)

        # Filter to only completed results for ranking
        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status != 'completed']

        # Sort completed results
        completed.sort(key=lambda r: getattr(r, by), reverse=not ascending)

        # Assign ranks
        for i, result in enumerate(completed, 1):
            result.rank = i

        # Failed results get rank 0
        for result in failed:
            result.rank = 0

        return completed + failed

    def format_table(
        self,
        results: List[VariationResult],
        top_n: Optional[int] = None,
        param_names: Optional[List[str]] = None
    ) -> str:
        """
        Format results as a CLI-friendly table.

        Args:
            results: Ranked results list
            top_n: Show only top N results
            param_names: Parameter names to show (auto-detect if None)
        """
        if not results:
            return "No results to display"

        # Determine which parameters to show
        if param_names is None:
            # Get all param names from first result
            param_names = list(results[0].params.keys()) if results else []

        # Limit results if requested
        display_results = results[:top_n] if top_n else results

        # Build header
        param_headers = [self._short_name(p) for p in param_names]
        headers = ['Rank'] + param_headers + ['Signals', 'Win%', 'Pips', 'PF', 'Score']

        # Calculate column widths
        widths = [len(h) for h in headers]

        # Build rows
        rows = []
        for result in display_results:
            if result.status != 'completed':
                continue

            param_values = [str(result.params.get(p, '-')) for p in param_names]
            row = [
                str(result.rank),
                *param_values,
                str(result.signal_count),
                f"{result.win_rate:.1f}%",
                f"{result.total_pips:+.1f}",
                f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "inf",
                f"{result.composite_score:.3f}",
            ]
            rows.append(row)

            # Update widths
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        # Format output
        lines = []

        # Header line
        header_line = ' | '.join(h.center(widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append('-' * len(header_line))

        # Data rows
        for row in rows:
            row_line = ' | '.join(val.rjust(widths[i]) for i, val in enumerate(row))
            lines.append(row_line)

        return '\n'.join(lines)

    def _short_name(self, param_name: str) -> str:
        """Convert parameter name to short display name"""
        shortcuts = {
            'fixed_stop_loss_pips': 'SL',
            'fixed_take_profit_pips': 'TP',
            'min_confidence': 'Conf',
            'max_confidence': 'MaxCf',
            'sl_buffer_pips': 'SlBuf',
            'min_risk_reward': 'RR',
            'ema_period': 'EMA',
            'swing_lookback_bars': 'Swing',
            'macd_filter_enabled': 'MACD',
            'cooldown_minutes': 'Cool',
            'fvg_minimum_size_pips': 'FVG',
        }
        return shortcuts.get(param_name, param_name[:6])

    def export_csv(self, results: List[VariationResult], filepath: str):
        """Export results to CSV file"""
        import csv

        if not results:
            logger.warning("No results to export")
            return

        # Get all param names
        param_names = list(results[0].params.keys()) if results else []

        headers = ['rank', 'variation_index'] + param_names + [
            'signal_count', 'win_count', 'loss_count', 'win_rate',
            'total_pips', 'profit_factor', 'composite_score',
            'avg_win_pips', 'avg_loss_pips', 'expectancy',
            'duration_seconds', 'status'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for result in results:
                row = [
                    result.rank,
                    result.variation_index,
                    *[result.params.get(p, '') for p in param_names],
                    result.signal_count,
                    result.win_count,
                    result.loss_count,
                    result.win_rate,
                    result.total_pips,
                    result.profit_factor if result.profit_factor != float('inf') else 'inf',
                    result.composite_score,
                    result.avg_win_pips,
                    result.avg_loss_pips,
                    result.expectancy,
                    result.duration_seconds,
                    result.status,
                ]
                writer.writerow(row)

        logger.info(f"Exported {len(results)} results to {filepath}")

    def summary(self, results: List[VariationResult]) -> str:
        """Generate summary statistics"""
        if not results:
            return "No results"

        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status != 'completed']

        lines = [
            f"Total variations: {len(results)}",
            f"Completed: {len(completed)}",
            f"Failed: {len(failed)}",
        ]

        if completed:
            best = completed[0]
            lines.extend([
                "",
                "Best parameters:",
                f"  {best.params}",
                f"  Win rate: {best.win_rate:.1f}%",
                f"  Total pips: {best.total_pips:+.1f}",
                f"  Composite score: {best.composite_score:.3f}",
            ])

        return '\n'.join(lines)
