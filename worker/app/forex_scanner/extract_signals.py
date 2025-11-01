#!/usr/bin/env python3
"""
Signal Data Extractor for Backtest Analysis

Extracts detailed signal data from backtest logs to perform evidence-based analysis.
This script parses backtest output to identify individual signals and their outcomes.
"""

import re
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

def parse_signal_from_log(signal_text: str) -> Dict[str, Any]:
    """
    Parse a single signal from log text

    Expected format:
    Signal #1 [BUY/SELL] - Timestamp - Epic - Price
    Confidence: XX%, Entry: XX.XXXX, SL: XX.XXXX, TP: XX.XXXX
    Result: WIN/LOSS/BREAKEVEN, Profit/Loss: XX.X pips
    """
    signal_data = {}

    lines = signal_text.strip().split('\n')

    # Parse header line
    header_match = re.search(r'Signal #(\d+) \[(\w+)\].*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', lines[0])
    if header_match:
        signal_data['signal_id'] = int(header_match.group(1))
        signal_data['direction'] = header_match.group(2)
        signal_data['timestamp'] = header_match.group(3)

    # Parse signal details
    for line in lines:
        # Confidence
        if 'Confidence:' in line:
            conf_match = re.search(r'Confidence:\s*(\d+\.?\d*)%', line)
            if conf_match:
                signal_data['confidence'] = float(conf_match.group(1))

        # Entry price
        if 'Entry:' in line:
            entry_match = re.search(r'Entry:\s*(\d+\.?\d*)', line)
            if entry_match:
                signal_data['entry_price'] = float(entry_match.group(1))

        # Stop loss
        if 'SL:' in line or 'Stop Loss:' in line:
            sl_match = re.search(r'S[Ll]:\s*(\d+\.?\d*)', line)
            if sl_match:
                signal_data['stop_loss'] = float(sl_match.group(1))

        # Take profit
        if 'TP:' in line or 'Take Profit:' in line:
            tp_match = re.search(r'T[Pp]:\s*(\d+\.?\d*)', line)
            if tp_match:
                signal_data['take_profit'] = float(tp_match.group(1))

        # Result
        if 'Result:' in line:
            result_match = re.search(r'Result:\s*(\w+)', line)
            if result_match:
                signal_data['result'] = result_match.group(1)

        # Profit/Loss pips
        if 'pips' in line.lower():
            pips_match = re.search(r'([-+]?\d+\.?\d*)\s*pips', line, re.IGNORECASE)
            if pips_match:
                signal_data['pips'] = float(pips_match.group(1))

    return signal_data


def extract_signals_from_file(log_file: str) -> List[Dict[str, Any]]:
    """Extract all signals from a backtest log file"""

    signals = []

    with open(log_file, 'r') as f:
        content = f.read()

    # Find signal sections (adjust pattern based on actual log format)
    signal_pattern = r'Signal #\d+.*?(?=Signal #\d+|\Z)'
    matches = re.findall(signal_pattern, content, re.DOTALL)

    for match in matches:
        signal = parse_signal_from_log(match)
        if signal:
            signals.append(signal)

    return signals


def categorize_signals(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Categorize signals into winners, losers, breakeven"""

    winners = []
    losers = []
    breakeven = []

    for signal in signals:
        result = signal.get('result', '').upper()

        if result == 'WIN' or result == 'WINNER':
            winners.append(signal)
        elif result == 'LOSS' or result == 'LOSER':
            losers.append(signal)
        elif result == 'BREAKEVEN' or result == 'BE':
            breakeven.append(signal)

    return {
        'total_signals': len(signals),
        'winners': winners,
        'losers': losers,
        'breakeven': breakeven,
        'win_rate': len(winners) / len(signals) * 100 if signals else 0,
        'loss_rate': len(losers) / len(signals) * 100 if signals else 0,
        'be_rate': len(breakeven) / len(signals) * 100 if signals else 0
    }


def analyze_losers(losers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze losing trades to identify patterns"""

    if not losers:
        return {}

    # Calculate statistics
    avg_confidence = sum(s.get('confidence', 0) for s in losers) / len(losers)
    avg_loss_pips = sum(abs(s.get('pips', 0)) for s in losers) / len(losers)

    # Direction distribution
    bull_losers = sum(1 for s in losers if s.get('direction') == 'BUY')
    bear_losers = sum(1 for s in losers if s.get('direction') == 'SELL')

    return {
        'total_losers': len(losers),
        'avg_confidence': avg_confidence,
        'avg_loss_pips': avg_loss_pips,
        'bull_losers': bull_losers,
        'bear_losers': bear_losers,
        'bull_loser_rate': bull_losers / len(losers) * 100,
        'bear_loser_rate': bear_losers / len(losers) * 100
    }


def main():
    """Main extraction and analysis function"""

    if len(sys.argv) < 2:
        print("Usage: python extract_signals.py <backtest_log_file>")
        print("Example: python extract_signals.py /tmp/backtest_output.log")
        sys.exit(1)

    log_file = sys.argv[1]

    print("🔍 Extracting signals from backtest log...")
    signals = extract_signals_from_file(log_file)

    if not signals:
        print("⚠️ No signals found in log file")
        print("Make sure you ran the backtest with --show-signals flag")
        sys.exit(1)

    print(f"✅ Found {len(signals)} signals")

    # Categorize signals
    categorized = categorize_signals(signals)

    print(f"\n📊 Signal Breakdown:")
    print(f"   Total: {categorized['total_signals']}")
    print(f"   Winners: {len(categorized['winners'])} ({categorized['win_rate']:.1f}%)")
    print(f"   Losers: {len(categorized['losers'])} ({categorized['loss_rate']:.1f}%)")
    print(f"   Breakeven: {len(categorized['breakeven'])} ({categorized['be_rate']:.1f}%)")

    # Analyze losers
    if categorized['losers']:
        loser_analysis = analyze_losers(categorized['losers'])
        print(f"\n📉 Loser Analysis:")
        print(f"   Total losers: {loser_analysis['total_losers']}")
        print(f"   Avg confidence: {loser_analysis['avg_confidence']:.1f}%")
        print(f"   Avg loss: {loser_analysis['avg_loss_pips']:.1f} pips")
        print(f"   BULL losers: {loser_analysis['bull_losers']} ({loser_analysis['bull_loser_rate']:.1f}%)")
        print(f"   BEAR losers: {loser_analysis['bear_losers']} ({loser_analysis['bear_loser_rate']:.1f}%)")

    # Export to JSON
    output_file = log_file.replace('.log', '_signals.json')
    with open(output_file, 'w') as f:
        json.dump({
            'signals': signals,
            'categorized': {
                'total': categorized['total_signals'],
                'winners': categorized['winners'],
                'losers': categorized['losers'],
                'breakeven': categorized['breakeven'],
                'statistics': {
                    'win_rate': categorized['win_rate'],
                    'loss_rate': categorized['loss_rate'],
                    'be_rate': categorized['be_rate']
                }
            }
        }, f, indent=2)

    print(f"\n💾 Exported detailed signal data to: {output_file}")


if __name__ == '__main__':
    main()
