"""
Signal Decision Logger for Backtest Analysis

Captures every signal evaluation (approved + rejected) with full decision context,
allowing detailed analysis of why signals pass or fail each filter stage.

Directory structure:
    logs/backtest_signals/
        execution_<id>/
            signal_decisions.csv          # All signal evaluations with filter values
            rejection_summary.json         # Aggregated rejection stats by reason
            backtest_summary.txt           # Full backtest summary (human-readable)
            backtest_summary.json          # Performance data (machine-readable)
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict


class SignalDecisionLogger:
    """
    Logs every signal evaluation during backtesting with complete decision context.

    Captures:
    - All filter values at each decision point
    - Rejection reason and step where rejection occurred
    - Approval with full metrics for approved signals
    - Backtest performance summary
    """

    # Standard rejection reasons (for consistency)
    REJECTION_REASONS = {
        'COOLDOWN_ACTIVE': 'Recent signal within cooldown period',
        'SESSION_FILTERED': 'Outside valid trading session',
        'HTF_MISALIGNMENT': 'Signal direction doesn\'t match HTF trend',
        'NO_BOS_CHOCH': 'BOS/CHoCH not detected on 15m',
        'LOW_BOS_QUALITY': 'BOS quality below threshold',
        'NO_ORDER_BLOCK': 'Order block not found',
        'NO_PATTERN': 'Rejection pattern not found',
        'WEAK_PATTERN': 'Pattern strength below minimum',
        'LOW_RR_RATIO': 'Risk/reward ratio below minimum',
        'LOW_CONFIDENCE': 'Overall confidence below minimum',
        'PREMIUM_DISCOUNT_REJECT': 'Wrong zone for entry direction',
        'EQUILIBRIUM_LOW_CONFIDENCE': 'In equilibrium with low confidence',
        'MOMENTUM_FILTER': 'Momentum filter rejection',
        'PIPELINE_VALIDATION': 'Failed pipeline validation'
    }

    def __init__(self, execution_id: int, log_dir: str = None):
        """
        Initialize signal decision logger.

        Args:
            execution_id: Backtest execution ID
            log_dir: Base log directory (default: logs/backtest_signals)
        """
        self.execution_id = execution_id

        # Set up directory structure
        if log_dir is None:
            base_dir = Path(__file__).parent.parent.parent / 'logs' / 'backtest_signals'
        else:
            base_dir = Path(log_dir)

        self.execution_dir = base_dir / f'execution_{execution_id}'
        self.execution_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.csv_path = self.execution_dir / 'signal_decisions.csv'
        self.rejection_summary_path = self.execution_dir / 'rejection_summary.json'
        self.summary_txt_path = self.execution_dir / 'backtest_summary.txt'
        self.summary_json_path = self.execution_dir / 'backtest_summary.json'

        # In-memory tracking
        self.decisions: List[Dict[str, Any]] = []
        self.rejection_counts = Counter()
        self.rejection_by_step = Counter()
        self.approved_count = 0
        self.rejected_count = 0

        # CSV file handle and writer
        self.csv_file = None
        self.csv_writer = None
        self._initialize_csv()

        # Backtest metadata
        self.backtest_results = None
        self.start_time = datetime.now()

    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file = open(self.csv_path, 'w', newline='')

        # Define comprehensive CSV headers
        headers = [
            # Basic info
            'timestamp', 'epic', 'pair', 'direction',

            # Decision
            'final_decision', 'rejection_reason', 'rejection_step',

            # Pre-checks
            'cooldown_active', 'session_valid',

            # HTF analysis
            'htf_trend', 'htf_strength', 'htf_structure', 'htf_in_pullback', 'htf_pullback_depth',

            # BOS/CHoCH
            'bos_detected', 'bos_direction', 'bos_quality',

            # Order Block
            'ob_found', 'ob_distance_pips',

            # Pattern
            'pattern_found', 'pattern_type', 'pattern_strength',

            # Support/Resistance
            'sr_level', 'sr_type', 'sr_strength', 'sr_distance_pips',

            # Premium/Discount
            'premium_discount_zone', 'entry_quality', 'zone_position_pct',

            # Risk/Reward
            'risk_pips', 'reward_pips', 'rr_ratio',

            # Confidence
            'confidence', 'htf_score', 'pattern_score', 'sr_score', 'rr_score',

            # Entry details
            'entry_price', 'stop_loss', 'take_profit'
        ]

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def log_signal_decision(
        self,
        timestamp: datetime,
        epic: str,
        pair: str,
        direction: str,
        decision: str,  # 'APPROVED' or 'REJECTED'
        rejection_reason: Optional[str] = None,
        rejection_step: Optional[str] = None,
        **filter_values
    ):
        """
        Log a signal evaluation decision with all filter values.

        Args:
            timestamp: Signal evaluation timestamp
            epic: Epic/symbol (e.g., 'CS.D.EURUSD')
            pair: Currency pair (e.g., 'EUR/USD')
            direction: 'bullish' or 'bearish'
            decision: 'APPROVED' or 'REJECTED'
            rejection_reason: Why signal was rejected (if rejected)
            rejection_step: Which step rejected the signal
            **filter_values: All filter values (htf_trend, bos_quality, etc.)
        """
        # Create decision record
        decision_record = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else timestamp,
            'epic': epic,
            'pair': pair,
            'direction': direction,
            'final_decision': decision,
            'rejection_reason': rejection_reason or '',
            'rejection_step': rejection_step or '',
        }

        # Add all filter values (with defaults for missing values)
        decision_record.update({
            'cooldown_active': filter_values.get('cooldown_active', ''),
            'session_valid': filter_values.get('session_valid', ''),
            'htf_trend': filter_values.get('htf_trend', ''),
            'htf_strength': filter_values.get('htf_strength', ''),
            'htf_structure': filter_values.get('htf_structure', ''),
            'htf_in_pullback': filter_values.get('htf_in_pullback', ''),
            'htf_pullback_depth': filter_values.get('htf_pullback_depth', ''),
            'bos_detected': filter_values.get('bos_detected', ''),
            'bos_direction': filter_values.get('bos_direction', ''),
            'bos_quality': filter_values.get('bos_quality', ''),
            'ob_found': filter_values.get('ob_found', ''),
            'ob_distance_pips': filter_values.get('ob_distance_pips', ''),
            'pattern_found': filter_values.get('pattern_found', ''),
            'pattern_type': filter_values.get('pattern_type', ''),
            'pattern_strength': filter_values.get('pattern_strength', ''),
            'sr_level': filter_values.get('sr_level', ''),
            'sr_type': filter_values.get('sr_type', ''),
            'sr_strength': filter_values.get('sr_strength', ''),
            'sr_distance_pips': filter_values.get('sr_distance_pips', ''),
            'premium_discount_zone': filter_values.get('premium_discount_zone', ''),
            'entry_quality': filter_values.get('entry_quality', ''),
            'zone_position_pct': filter_values.get('zone_position_pct', ''),
            'risk_pips': filter_values.get('risk_pips', ''),
            'reward_pips': filter_values.get('reward_pips', ''),
            'rr_ratio': filter_values.get('rr_ratio', ''),
            'confidence': filter_values.get('confidence', ''),
            'htf_score': filter_values.get('htf_score', ''),
            'pattern_score': filter_values.get('pattern_score', ''),
            'sr_score': filter_values.get('sr_score', ''),
            'rr_score': filter_values.get('rr_score', ''),
            'entry_price': filter_values.get('entry_price', ''),
            'stop_loss': filter_values.get('stop_loss', ''),
            'take_profit': filter_values.get('take_profit', ''),
        })

        # Write to CSV immediately
        self.csv_writer.writerow(decision_record)
        self.csv_file.flush()

        # Update in-memory tracking
        self.decisions.append(decision_record)

        if decision == 'APPROVED':
            self.approved_count += 1
        else:
            self.rejected_count += 1
            if rejection_reason:
                self.rejection_counts[rejection_reason] += 1
            if rejection_step:
                self.rejection_by_step[rejection_step] += 1

    def set_backtest_results(self, results: Dict[str, Any]):
        """
        Store backtest performance results for summary generation.

        Args:
            results: Dict with backtest performance data (wins, losses, profit factor, etc.)
        """
        self.backtest_results = results

    def finalize(self):
        """
        Finalize logging: close CSV, generate summaries.
        Call this at the end of the backtest.
        """
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()

        # Generate rejection summary JSON
        self._generate_rejection_summary()

        # Generate backtest summary (if results provided)
        if self.backtest_results:
            self._generate_backtest_summary()

    def _generate_rejection_summary(self):
        """Generate rejection breakdown JSON."""
        total_evaluations = self.approved_count + self.rejected_count

        summary = {
            'execution_id': self.execution_id,
            'total_evaluations': total_evaluations,
            'approved': self.approved_count,
            'rejected': self.rejected_count,
            'approval_rate': round(self.approved_count / total_evaluations, 4) if total_evaluations > 0 else 0,
            'rejection_breakdown': dict(self.rejection_counts),
            'rejection_by_step': dict(self.rejection_by_step),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.rejection_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_backtest_summary(self):
        """Generate human-readable and JSON backtest summaries."""
        if not self.backtest_results:
            return

        results = self.backtest_results
        total_evaluations = self.approved_count + self.rejected_count

        # Generate TXT summary (human-readable)
        summary_txt = self._format_text_summary(results, total_evaluations)
        with open(self.summary_txt_path, 'w') as f:
            f.write(summary_txt)

        # Generate JSON summary (machine-readable)
        summary_json = {
            'execution_id': self.execution_id,
            'performance': results,
            'signal_analysis': {
                'total_evaluations': total_evaluations,
                'approved': self.approved_count,
                'rejected': self.rejected_count,
                'approval_rate': round(self.approved_count / total_evaluations, 4) if total_evaluations > 0 else 0,
                'top_rejection_reasons': dict(self.rejection_counts.most_common(5))
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.summary_json_path, 'w') as f:
            json.dump(summary_json, f, indent=2)

    def _format_text_summary(self, results: Dict[str, Any], total_evaluations: int) -> str:
        """Format human-readable text summary."""
        lines = []
        lines.append("=" * 80)
        lines.append(" " * 20 + "BACKTEST RESULTS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Execution ID: {self.execution_id}")
        lines.append(f"Strategy: {results.get('strategy', 'N/A')}")
        lines.append(f"Period: {results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')} ({results.get('days', 'N/A')} days)")
        lines.append(f"Pairs: {results.get('pairs', 'N/A')}")
        lines.append(f"Timeframe: {results.get('timeframe', 'N/A')}")
        lines.append("")

        lines.append("PERFORMANCE METRICS")
        lines.append("=" * 80)
        lines.append(f"Total Signals: {results.get('total_signals', 0)}")
        lines.append(f"Winning Trades: {results.get('winning_trades', 0)} ({results.get('win_rate', 0):.1f}%)")
        lines.append(f"Losing Trades: {results.get('losing_trades', 0)} ({100 - results.get('win_rate', 0):.1f}%)")
        lines.append("")
        lines.append(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        lines.append(f"Expectancy: {results.get('expectancy', 0):+.1f} pips per trade")
        lines.append("")
        lines.append(f"Average Win: {results.get('avg_win', 0):.1f} pips")
        lines.append(f"Average Loss: {results.get('avg_loss', 0):.1f} pips")
        lines.append(f"Risk/Reward Ratio: {results.get('rr_ratio', 0):.2f}:1")
        lines.append("")
        lines.append(f"Largest Win: {results.get('largest_win', 0):.1f} pips")
        lines.append(f"Largest Loss: {results.get('largest_loss', 0):.1f} pips")
        lines.append("")

        lines.append("SIGNAL DECISION BREAKDOWN")
        lines.append("=" * 80)
        lines.append(f"Total Evaluations: {total_evaluations:,}")
        lines.append(f"Signals Generated: {self.approved_count} ({self.approved_count/total_evaluations*100:.1f}% approval rate)")
        lines.append(f"Signals Rejected: {self.rejected_count} ({self.rejected_count/total_evaluations*100:.1f}%)")
        lines.append("")

        lines.append("TOP REJECTION REASONS")
        lines.append("=" * 80)
        for i, (reason, count) in enumerate(self.rejection_counts.most_common(10), 1):
            pct = count / self.rejected_count * 100 if self.rejected_count > 0 else 0
            lines.append(f"{i}. {reason}: {count:,} ({pct:.1f}%)")
        lines.append("")

        lines.append("MONTHLY PERFORMANCE")
        lines.append("=" * 80)
        lines.append(f"Gross Profit: +{results.get('gross_profit', 0):.1f} pips")
        lines.append(f"Gross Loss: {results.get('gross_loss', 0):.1f} pips")
        lines.append(f"Net Profit: {results.get('net_profit', 0):+.1f} pips")
        lines.append("")

        lines.append("FILES GENERATED")
        lines.append("=" * 80)
        lines.append(f"📋 Signal Decisions: signal_decisions.csv ({total_evaluations:,} rows)")
        lines.append(f"📈 Rejection Summary: rejection_summary.json")
        lines.append(f"📊 Performance Data: backtest_summary.json")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def get_log_directory(self) -> str:
        """Get the execution log directory path."""
        return str(self.execution_dir)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.finalize()
