#!/usr/bin/env python3
"""
Analyze MACD signal rejections from forex scanner logs
Finds the most common reasons why MACD signals are blocked
"""

import re
from collections import Counter, defaultdict
from datetime import datetime
import sys

def parse_log_file(log_path):
    """Parse log file and extract MACD rejection reasons"""

    rejections = []
    rejection_patterns = [
        # Swing proximity rejections
        (r'❌ (BULL|BEAR|ADX BULL|ADX BEAR) rejected: (.+?)(?:\n|$)', 'swing'),

        # ADX rejections
        (r'❌.*?(BULL|BEAR) rejected: ADX ([\d.]+) < ([\d.]+)', 'adx_threshold'),
        (r'❌.*?(BULL|BEAR) rejected: Invalid ADX', 'invalid_adx'),

        # Histogram rejections
        (r'❌.*?(BULL|BEAR) rejected: Histogram ([\d.]+) too small.*?min=([\d.]+)', 'histogram_too_small'),
        (r'❌.*?(BULL|BEAR) rejected: negative histogram', 'histogram_wrong_direction'),
        (r'❌.*?(BULL|BEAR) rejected: positive histogram', 'histogram_wrong_direction'),

        # MACD line rejections
        (r'❌.*?(BULL|BEAR) rejected: MACD line too (positive|negative) ([\d.]+)', 'macd_line_position'),

        # RSI rejections
        (r'❌.*?(BULL|BEAR) rejected: RSI ([\d.]+) (overbought|oversold)', 'rsi_extreme'),

        # EMA filter rejections
        (r'❌.*?(BULL|BEAR) rejected: price.*?(below|above) EMA', 'ema_filter'),

        # Confidence rejections
        (r'❌.*?(BULL|BEAR) rejected: confidence ([\d.]+)% < ([\d.]+)%', 'low_confidence'),
    ]

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Check each rejection pattern
                for pattern, reason_type in rejection_patterns:
                    match = re.search(pattern, line)
                    if match:
                        signal_type = match.group(1)

                        # Extract full reason based on type
                        if reason_type == 'swing':
                            full_reason = match.group(2).strip()
                            rejections.append({
                                'type': reason_type,
                                'signal': signal_type,
                                'reason': full_reason,
                                'line': line.strip()
                            })
                        elif reason_type == 'adx_threshold':
                            adx_value = match.group(2)
                            threshold = match.group(3)
                            rejections.append({
                                'type': reason_type,
                                'signal': signal_type,
                                'reason': f'ADX {adx_value} < {threshold}',
                                'line': line.strip()
                            })
                        elif reason_type == 'histogram_too_small':
                            hist_value = match.group(2)
                            min_value = match.group(3)
                            rejections.append({
                                'type': reason_type,
                                'signal': signal_type,
                                'reason': f'Histogram {hist_value} < minimum {min_value}',
                                'line': line.strip()
                            })
                        else:
                            rejections.append({
                                'type': reason_type,
                                'signal': signal_type,
                                'reason': reason_type.replace('_', ' ').title(),
                                'line': line.strip()
                            })
                        break
    except Exception as e:
        print(f"Error reading {log_path}: {e}")

    return rejections


def analyze_rejections(rejections):
    """Analyze and categorize rejections"""

    # Count by type
    type_counter = Counter([r['type'] for r in rejections])

    # Count by signal direction
    signal_counter = Counter([r['signal'] for r in rejections])

    # Count specific reasons
    reason_counter = Counter([r['reason'] for r in rejections])

    # Group by type and signal
    type_signal_counter = Counter([(r['type'], r['signal']) for r in rejections])

    return {
        'total': len(rejections),
        'by_type': type_counter,
        'by_signal': signal_counter,
        'by_reason': reason_counter,
        'by_type_signal': type_signal_counter,
        'raw_rejections': rejections
    }


def print_analysis(analysis):
    """Print formatted analysis results"""

    print("=" * 80)
    print("MACD SIGNAL REJECTION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal rejections found: {analysis['total']}")

    if analysis['total'] == 0:
        print("\nNo rejections found in log files.")
        return

    # Print by rejection type
    print("\n" + "=" * 80)
    print("REJECTIONS BY TYPE")
    print("=" * 80)
    for reason_type, count in analysis['by_type'].most_common():
        percentage = (count / analysis['total']) * 100
        print(f"{reason_type:25} {count:6} ({percentage:5.1f}%)")

    # Print by signal direction
    print("\n" + "=" * 80)
    print("REJECTIONS BY SIGNAL DIRECTION")
    print("=" * 80)
    for signal, count in analysis['by_signal'].most_common():
        percentage = (count / analysis['total']) * 100
        print(f"{signal:15} {count:6} ({percentage:5.1f}%)")

    # Print top specific reasons
    print("\n" + "=" * 80)
    print("TOP 20 SPECIFIC REJECTION REASONS")
    print("=" * 80)
    for reason, count in analysis['by_reason'].most_common(20):
        percentage = (count / analysis['total']) * 100
        print(f"{count:6} ({percentage:5.1f}%) - {reason}")

    # Print breakdown by type and signal
    print("\n" + "=" * 80)
    print("BREAKDOWN BY TYPE AND SIGNAL")
    print("=" * 80)
    for (reason_type, signal), count in sorted(analysis['by_type_signal'].items(),
                                               key=lambda x: x[1], reverse=True):
        percentage = (count / analysis['total']) * 100
        print(f"{reason_type:25} - {signal:15} {count:6} ({percentage:5.1f}%)")

    # Print sample rejections for top types
    print("\n" + "=" * 80)
    print("SAMPLE REJECTIONS (Top 3 Categories)")
    print("=" * 80)

    top_types = [t for t, _ in analysis['by_type'].most_common(3)]
    for reason_type in top_types:
        print(f"\n--- {reason_type.upper()} (Sample entries) ---")
        samples = [r for r in analysis['raw_rejections'] if r['type'] == reason_type][:3]
        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. {sample['signal']}: {sample['reason']}")
            # Print truncated log line
            log_line = sample['line'][:150] + "..." if len(sample['line']) > 150 else sample['line']
            print(f"   {log_line}")


def main():
    """Main analysis function"""

    # Log files to analyze
    log_files = [
        '/home/hr/Projects/TradeSystemV1/logs/worker/forex_scanner.log',
        '/home/hr/Projects/TradeSystemV1/logs/worker/forex_scanner.2025-10-13.log',
        '/home/hr/Projects/TradeSystemV1/logs/worker/forex_scanner.2025-10-12.log',
        '/home/hr/Projects/TradeSystemV1/logs/worker/forex_scanner.2025-10-11.log',
    ]

    # Allow custom log file from command line
    if len(sys.argv) > 1:
        log_files = sys.argv[1:]

    print(f"Analyzing {len(log_files)} log file(s)...")
    print("Log files:")
    for log_file in log_files:
        print(f"  - {log_file}")

    # Parse all log files
    all_rejections = []
    for log_file in log_files:
        print(f"\nParsing {log_file}...")
        rejections = parse_log_file(log_file)
        print(f"  Found {len(rejections)} rejections")
        all_rejections.extend(rejections)

    # Analyze rejections
    analysis = analyze_rejections(all_rejections)

    # Print results
    print_analysis(analysis)


if __name__ == '__main__':
    main()
