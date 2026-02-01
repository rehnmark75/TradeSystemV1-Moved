#!/usr/bin/env python3
"""
Regime Optimization Runner

Runs parameter grid search backtests during regime-dominated periods to find
optimal trading parameters for each market regime.

This script:
1. Loads regime periods from regime_period_analyzer (or accepts manual dates)
2. Runs backtests with parameter combinations for each period
3. Collects and ranks results by performance metrics
4. Outputs optimal parameters per regime

Usage:
    # Run parameter grid for trending periods
    python regime_optimization_runner.py --epic EURUSD --regime trending --days 180

    # Run with custom parameter grid from JSON file
    python regime_optimization_runner.py --epic EURUSD --regime high_volatility --grid params.json

    # Run for specific date range
    python regime_optimization_runner.py --epic EURUSD --start-date 2025-11-01 --end-date 2025-11-15

    # Dry run - show what would be tested
    python regime_optimization_runner.py --epic EURUSD --regime trending --dry-run

Parameter Grid:
    Default grid tests combinations of:
    - fixed_stop_loss_pips: 8-18 (based on regime volatility)
    - fixed_take_profit_pips: 8-25 (based on regime trends)
    - min_confidence: 0.40-0.65 (signal quality)
    - macd_filter_enabled: true/false
"""

import argparse
import json
import os
import subprocess
import sys
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import itertools

# Default parameter grids per regime
DEFAULT_PARAM_GRIDS = {
    'trending': {
        'fixed_stop_loss_pips': [8, 10, 12],
        'fixed_take_profit_pips': [15, 18, 20, 25],
        'min_confidence': [0.40, 0.45, 0.50, 0.55],
        'macd_filter_enabled': [True, False]
    },
    'ranging': {
        'fixed_stop_loss_pips': [10, 12, 15],
        'fixed_take_profit_pips': [10, 12, 15],
        'min_confidence': [0.50, 0.55, 0.60, 0.65],
        'macd_filter_enabled': [True]
    },
    'high_volatility': {
        'fixed_stop_loss_pips': [12, 15, 18],
        'fixed_take_profit_pips': [8, 10, 12],
        'min_confidence': [0.55, 0.60, 0.65, 0.70],
        'macd_filter_enabled': [True]
    },
    'low_volatility': {
        'fixed_stop_loss_pips': [8, 10, 12],
        'fixed_take_profit_pips': [12, 15, 18],
        'min_confidence': [0.45, 0.50, 0.55],
        'macd_filter_enabled': [False, True]
    },
    'breakout': {
        'fixed_stop_loss_pips': [10, 12, 15],
        'fixed_take_profit_pips': [15, 20, 25],
        'min_confidence': [0.50, 0.55, 0.60],
        'macd_filter_enabled': [True, False]
    },
    'reversal': {
        'fixed_stop_loss_pips': [10, 12, 15],
        'fixed_take_profit_pips': [12, 15, 18],
        'min_confidence': [0.55, 0.60, 0.65],
        'macd_filter_enabled': [True]
    }
}


@dataclass
class BacktestResult:
    """Result from a single backtest run"""
    params: Dict[str, Any]
    total_signals: int = 0
    total_pips: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pips: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    success: bool = True
    error: Optional[str] = None
    raw_output: str = ""


@dataclass
class OptimizationRun:
    """A complete optimization run for a regime period"""
    epic: str
    regime: str
    start_date: str
    end_date: str
    param_grid: Dict[str, List]
    results: List[BacktestResult] = field(default_factory=list)
    best_result: Optional[BacktestResult] = None
    total_combinations: int = 0
    completed_combinations: int = 0


def normalize_epic(epic: str) -> str:
    """Convert epic shorthand to full format"""
    if epic.startswith('CS.D.'):
        return epic

    epic = epic.upper()
    # EURUSD uses CEEM, all others use MINI
    if epic == 'EURUSD':
        return f'CS.D.{epic}.CEEM.IP'
    return f'CS.D.{epic}.MINI.IP'


def epic_shorthand(epic: str) -> str:
    """Convert full epic to shorthand"""
    if epic.startswith('CS.D.'):
        parts = epic.split('.')
        if len(parts) >= 3:
            return parts[2]
    return epic.upper()


def build_override_args(params: Dict[str, Any]) -> List[str]:
    """Build --override arguments from parameter dict"""
    args = []
    for key, value in params.items():
        if isinstance(value, bool):
            value_str = 'true' if value else 'false'
        else:
            value_str = str(value)
        args.extend(['--override', f'{key}={value_str}'])
    return args


def run_backtest(
    epic: str,
    start_date: str,
    end_date: str,
    params: Dict[str, Any],
    scalp_mode: bool = True,
    use_historical_intelligence: bool = True,
    verbose: bool = False
) -> BacktestResult:
    """
    Run a single backtest with specified parameters.

    Args:
        epic: Currency pair (full or shorthand)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        params: Parameter overrides
        scalp_mode: Enable scalp mode (default True)
        use_historical_intelligence: Use historical market intelligence
        verbose: Show backtest output

    Returns:
        BacktestResult with metrics
    """
    result = BacktestResult(params=params)

    try:
        # Build command
        cmd = [
            'python3', '-u', '/app/forex_scanner/backtest_cli.py',
            '--epic', normalize_epic(epic),
            '--start-date', start_date,
            '--end-date', end_date
        ]

        if scalp_mode:
            cmd.append('--scalp')

        if use_historical_intelligence:
            cmd.append('--use-historical-intelligence')
        else:
            cmd.append('--no-historical-intelligence')

        # Add parameter overrides
        cmd.extend(build_override_args(params))

        if verbose:
            print(f"\n  Running: {' '.join(cmd)}")

        # Run backtest
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        output = process.stdout + process.stderr
        result.raw_output = output

        if process.returncode != 0:
            result.success = False
            result.error = f"Backtest failed with return code {process.returncode}"
            return result

        # Parse results from output
        result = parse_backtest_output(output, params)

    except subprocess.TimeoutExpired:
        result.success = False
        result.error = "Backtest timed out (5 minutes)"
    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


def parse_backtest_output(output: str, params: Dict[str, Any]) -> BacktestResult:
    """Parse backtest CLI output to extract metrics"""
    result = BacktestResult(params=params)
    result.raw_output = output

    try:
        # Pattern matching for metrics in backtest output
        # Output uses emoji prefixes and the format:
        #    📊 Total Signals: 12
        #    🎯 Win Rate: 91.7%
        #    📊 Profit Factor: 27.50
        #    💵 Expectancy: 22.5 pips per trade
        #    💰 Average Profit per Winner: 25.5 pips
        #    📉 Average Loss per Loser: 10.2 pips
        #    ✅ Winners: 11
        #    ❌ Losers: 1

        # Total signals
        match = re.search(r'Total Signals:\s*(\d+)', output)
        if match:
            result.total_signals = int(match.group(1))

        # Win rate - use the last occurrence (from strategy performance section)
        matches = re.findall(r'Win Rate:\s*([\d.]+)%', output)
        if matches:
            result.win_rate = float(matches[-1])

        # Profit factor
        match = re.search(r'Profit Factor:\s*([\d.]+)', output)
        if match:
            result.profit_factor = float(match.group(1))

        # Expectancy (in pips per trade)
        match = re.search(r'Expectancy:\s*([+-]?[\d.]+)\s*pips', output)
        if match:
            result.expectancy = float(match.group(1))

        # Average profit per winner (with emoji)
        match = re.search(r'Average Profit per Winner:\s*([\d.]+)\s*pips', output)
        if match:
            result.avg_win_pips = float(match.group(1))

        # Average loss per loser (with emoji)
        match = re.search(r'Average Loss per Loser:\s*([\d.]+)\s*pips', output)
        if match:
            result.avg_loss_pips = float(match.group(1))

        # Winners count - use last occurrence (from strategy performance section)
        matches = re.findall(r'Winners:\s*(\d+)', output)
        if matches:
            result.winning_trades = int(matches[-1])

        # Losers count - use last occurrence
        matches = re.findall(r'Losers:\s*(\d+)', output)
        if matches:
            result.losing_trades = int(matches[-1])

        # Calculate total pips from expectancy if not directly available
        if result.expectancy != 0 and result.total_signals > 0:
            result.total_pips = result.expectancy * result.total_signals

        # Check if we got any meaningful results
        result.success = result.total_signals > 0

    except Exception as e:
        result.success = False
        result.error = f"Failed to parse output: {e}"

    return result


def get_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations from parameter grid"""
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_optimization(
    epic: str,
    start_date: str,
    end_date: str,
    regime: str,
    param_grid: Optional[Dict[str, List]] = None,
    scalp_mode: bool = True,
    use_historical_intelligence: bool = True,
    verbose: bool = False,
    max_combinations: int = 100
) -> OptimizationRun:
    """
    Run parameter optimization for a regime period.

    Args:
        epic: Currency pair
        start_date: Period start (YYYY-MM-DD)
        end_date: Period end (YYYY-MM-DD)
        regime: Market regime name
        param_grid: Custom parameter grid (None = use default for regime)
        scalp_mode: Enable scalp mode
        use_historical_intelligence: Use historical market intelligence
        verbose: Show detailed output
        max_combinations: Maximum combinations to test

    Returns:
        OptimizationRun with all results
    """
    # Use default grid if not provided
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS.get(regime, DEFAULT_PARAM_GRIDS['trending'])

    combinations = get_param_combinations(param_grid)
    total = len(combinations)

    if total > max_combinations:
        print(f"  Warning: {total} combinations exceeds limit ({max_combinations}). Truncating.")
        combinations = combinations[:max_combinations]

    run = OptimizationRun(
        epic=epic_shorthand(epic),
        regime=regime,
        start_date=start_date,
        end_date=end_date,
        param_grid=param_grid,
        total_combinations=len(combinations)
    )

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION: {run.epic} - {regime.upper()}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Testing {len(combinations)} parameter combinations")
    print(f"{'='*70}")

    for i, params in enumerate(combinations, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n[{i}/{len(combinations)}] Testing: {param_str}")

        result = run_backtest(
            epic=epic,
            start_date=start_date,
            end_date=end_date,
            params=params,
            scalp_mode=scalp_mode,
            use_historical_intelligence=use_historical_intelligence,
            verbose=verbose
        )

        run.results.append(result)
        run.completed_combinations += 1

        if result.success:
            print(f"  Signals: {result.total_signals}, Pips: {result.total_pips:+.1f}, "
                  f"WR: {result.win_rate:.1f}%, PF: {result.profit_factor:.2f}")
        else:
            print(f"  Failed: {result.error}")

    # Find best result
    valid_results = [r for r in run.results if r.success and r.total_signals >= 5]
    if valid_results:
        # Rank by profit factor, then win rate
        run.best_result = max(valid_results, key=lambda r: (r.profit_factor, r.win_rate))

        print(f"\n{'='*70}")
        print(f"BEST RESULT for {regime.upper()}")
        print(f"{'='*70}")
        print(f"Parameters: {run.best_result.params}")
        print(f"Signals: {run.best_result.total_signals}")
        print(f"Total Pips: {run.best_result.total_pips:+.1f}")
        print(f"Win Rate: {run.best_result.win_rate:.1f}%")
        print(f"Profit Factor: {run.best_result.profit_factor:.2f}")
        print(f"Expectancy: {run.best_result.expectancy:+.2f} pips/trade")

    return run


def load_regime_periods(
    epic: str,
    regime: str,
    days: int = 180,
    min_pct: float = 60.0
) -> List[Dict]:
    """
    Load regime-dominated periods from the analyzer.

    Args:
        epic: Currency pair
        regime: Target regime
        days: Days to analyze
        min_pct: Minimum regime percentage

    Returns:
        List of period dicts with start_date, end_date, regime_pct
    """
    try:
        # Run the analyzer to get periods
        cmd = [
            'python3', '/app/forex_scanner/scripts/regime_period_analyzer.py',
            '--epic', epic,
            '--regime', regime,
            '--days', str(days),
            '--min-pct', str(min_pct),
            '--output-json'
        ]

        process = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if process.returncode != 0:
            print(f"Warning: Failed to load regime periods: {process.stderr}")
            return []

        data = json.loads(process.stdout)
        return data.get('dominated_periods', [])

    except Exception as e:
        print(f"Warning: Could not load regime periods: {e}")
        return []


def aggregate_results(runs: List[OptimizationRun]) -> Dict:
    """Aggregate results from multiple optimization runs"""
    summary = {
        'epic': runs[0].epic if runs else None,
        'total_runs': len(runs),
        'regime_results': {}
    }

    for run in runs:
        if run.best_result:
            summary['regime_results'][run.regime] = {
                'period': f"{run.start_date} to {run.end_date}",
                'best_params': run.best_result.params,
                'signals': run.best_result.total_signals,
                'total_pips': run.best_result.total_pips,
                'win_rate': run.best_result.win_rate,
                'profit_factor': run.best_result.profit_factor,
                'expectancy': run.best_result.expectancy,
                'combinations_tested': run.completed_combinations
            }

    return summary


def print_summary(summary: Dict):
    """Print optimization summary"""
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION SUMMARY: {summary['epic']}")
    print(f"{'='*70}")

    for regime, result in summary['regime_results'].items():
        print(f"\n{regime.upper()}:")
        print(f"  Period: {result['period']}")
        print(f"  Best Parameters:")
        for k, v in result['best_params'].items():
            print(f"    - {k}: {v}")
        print(f"  Performance:")
        print(f"    - Signals: {result['signals']}")
        print(f"    - Total Pips: {result['total_pips']:+.1f}")
        print(f"    - Win Rate: {result['win_rate']:.1f}%")
        print(f"    - Profit Factor: {result['profit_factor']:.2f}")
        print(f"    - Expectancy: {result['expectancy']:+.2f} pips/trade")


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter optimization for market regime periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize for trending periods
  python regime_optimization_runner.py --epic EURUSD --regime trending

  # Optimize for specific date range
  python regime_optimization_runner.py --epic EURUSD \\
      --start-date 2025-11-01 --end-date 2025-11-15 --regime trending

  # Use custom parameter grid
  python regime_optimization_runner.py --epic EURUSD --regime high_volatility \\
      --grid '{"fixed_stop_loss_pips": [10, 12, 15], "min_confidence": [0.55, 0.60]}'

  # Dry run to see combinations
  python regime_optimization_runner.py --epic EURUSD --regime trending --dry-run
        """
    )

    parser.add_argument('--epic', type=str, default='EURUSD',
                        help='Currency pair (default: EURUSD)')
    parser.add_argument('--regime', type=str, required=True,
                        choices=['trending', 'ranging', 'high_volatility',
                                 'low_volatility', 'breakout', 'reversal'],
                        help='Market regime to optimize for')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD). If not provided, uses analyzer')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD). Required if --start-date provided')
    parser.add_argument('--days', type=int, default=180,
                        help='Days to search for regime periods (default: 180)')
    parser.add_argument('--min-pct', type=float, default=60.0,
                        help='Minimum regime percentage for periods (default: 60)')
    parser.add_argument('--grid', type=str,
                        help='Custom parameter grid as JSON string or file path')
    parser.add_argument('--max-combinations', type=int, default=100,
                        help='Maximum parameter combinations to test (default: 100)')
    parser.add_argument('--no-scalp', action='store_true',
                        help='Disable scalp mode')
    parser.add_argument('--no-historical-intelligence', action='store_true',
                        help='Disable historical market intelligence')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be tested without running')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Load custom grid if provided
    param_grid = None
    if args.grid:
        if os.path.isfile(args.grid):
            with open(args.grid) as f:
                param_grid = json.load(f)
        else:
            param_grid = json.loads(args.grid)

    # Get period(s) to test
    periods = []
    if args.start_date and args.end_date:
        # Manual date range
        periods = [{
            'start_date': args.start_date,
            'end_date': args.end_date,
            'regime': args.regime,
            'regime_pct': 100.0,
            'days': (datetime.strptime(args.end_date, '%Y-%m-%d') -
                     datetime.strptime(args.start_date, '%Y-%m-%d')).days
        }]
    else:
        # Load from analyzer
        periods = load_regime_periods(
            epic=args.epic,
            regime=args.regime,
            days=args.days,
            min_pct=args.min_pct
        )

    if not periods:
        print(f"No {args.regime} periods found. Try lowering --min-pct or using --start-date/--end-date")
        sys.exit(1)

    # Show what will be tested
    print(f"\nFound {len(periods)} {args.regime} period(s) to optimize:")
    for p in periods:
        print(f"  - {p['start_date']} to {p['end_date']} ({p['days']} days, {p['regime_pct']:.0f}% {args.regime})")

    # Use default grid if not provided
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRIDS.get(args.regime, DEFAULT_PARAM_GRIDS['trending'])

    combinations = get_param_combinations(param_grid)
    print(f"\nParameter grid: {len(combinations)} combinations")
    for key, values in param_grid.items():
        print(f"  - {key}: {values}")

    if args.dry_run:
        print("\n[DRY RUN] Would test the following combinations:")
        for i, params in enumerate(combinations[:10], 1):
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {i}. {param_str}")
        if len(combinations) > 10:
            print(f"  ... and {len(combinations) - 10} more")
        sys.exit(0)

    # Run optimization for each period
    all_runs = []
    for period in periods:
        run = run_optimization(
            epic=args.epic,
            start_date=period['start_date'],
            end_date=period['end_date'],
            regime=args.regime,
            param_grid=param_grid,
            scalp_mode=not args.no_scalp,
            use_historical_intelligence=not args.no_historical_intelligence,
            verbose=args.verbose,
            max_combinations=args.max_combinations
        )
        all_runs.append(run)

    # Aggregate and display results
    summary = aggregate_results(all_runs)

    if args.output_json:
        # Convert to JSON-serializable format
        output = {
            'epic': summary['epic'],
            'regime': args.regime,
            'total_runs': summary['total_runs'],
            'results': []
        }
        for run in all_runs:
            run_data = {
                'period': f"{run.start_date} to {run.end_date}",
                'total_combinations': run.total_combinations,
                'completed': run.completed_combinations,
                'results': []
            }
            for r in run.results:
                run_data['results'].append({
                    'params': r.params,
                    'success': r.success,
                    'signals': r.total_signals,
                    'total_pips': r.total_pips,
                    'win_rate': r.win_rate,
                    'profit_factor': r.profit_factor,
                    'expectancy': r.expectancy
                })
            if run.best_result:
                run_data['best'] = {
                    'params': run.best_result.params,
                    'signals': run.best_result.total_signals,
                    'total_pips': run.best_result.total_pips,
                    'win_rate': run.best_result.win_rate,
                    'profit_factor': run.best_result.profit_factor,
                    'expectancy': run.best_result.expectancy
                }
            output['results'].append(run_data)

        print(json.dumps(output, indent=2))
    else:
        print_summary(summary)


if __name__ == '__main__':
    main()
