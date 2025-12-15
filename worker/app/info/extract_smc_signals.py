#!/usr/bin/env python3
"""
Extract SMC signals from backtest log with proper parsing
"""

import re
import json
from collections import defaultdict

def extract_signals(log_file):
    """Extract all signals from log file"""
    signals = []

    with open(log_file, 'r') as f:
        content = f.read()

    # Split by signal detection markers
    signal_blocks = re.split(r'✅ \[SMC_STRUCTURE\] Signal detected for', content)

    print(f"Found {len(signal_blocks)-1} signal blocks")

    for i, block in enumerate(signal_blocks[1:], 1):  # Skip first empty split
        signal = {'signal_id': i}

        # Extract epic and direction from first line
        first_line_match = re.search(r'(CS\.D\.[\w.]+):\s*(BULL|BEAR)\s*@\s*([\d.]+)', block)
        if first_line_match:
            signal['epic'] = first_line_match.group(1)
            signal['direction'] = first_line_match.group(2)
            signal['entry_price'] = float(first_line_match.group(3))

            # Extract pair name
            pair_match = re.search(r'CS\.D\.(\w+)\.', signal['epic'])
            if pair_match:
                signal['pair'] = pair_match.group(1)

        # Look backward in previous block for signal details
        prev_block_start = max(0, content.find(signal_blocks[i]) - 10000)
        prev_block = content[prev_block_start:content.find(signal_blocks[i])]

        # Extract HTF trend strength
        htf_match = re.search(r'HTF Trend confirmed:\s*(BULL|BEAR|RANGING)\s*\(strength:\s*([\d.]+)%\)', prev_block)
        if htf_match:
            signal['htf_trend'] = htf_match.group(1)
            signal['htf_strength'] = float(htf_match.group(2))

        # Extract zone info
        zone_match = re.search(r'📍 Current Zone:\s*(PREMIUM|DISCOUNT|EQUILIBRIUM)', prev_block)
        if zone_match:
            signal['entry_zone'] = zone_match.group(1)

        # Extract price position
        pos_match = re.search(r'📈 Price Position:\s*([\d.]+)%', prev_block)
        if pos_match:
            signal['price_position'] = float(pos_match.group(1))

        # Extract zone validation result
        if 'entry in' in prev_block and 'zone' in prev_block:
            if '✅' in prev_block and ('TREND CONTINUATION' in prev_block or 'excellent timing' in prev_block):
                signal['zone_validation'] = 'PASS'
                if 'TREND CONTINUATION' in prev_block:
                    signal['zone_reason'] = 'TREND_CONTINUATION'
                elif 'excellent timing' in prev_block:
                    signal['zone_reason'] = 'EXCELLENT_TIMING'
            elif '❌' in prev_block and 'poor timing' in prev_block:
                signal['zone_validation'] = 'FAIL'
                signal['zone_reason'] = 'POOR_TIMING'

        # Extract R:R ratio
        rr_match = re.search(r'R:R Ratio:\s*([\d.]+)', prev_block)
        if rr_match:
            signal['rr_ratio'] = float(rr_match.group(1))

        # Extract stop loss and take profit
        sl_match = re.search(r'Stop Loss:\s*([\d.]+)\s*\(([\d.]+)\s*pips\)', prev_block)
        if sl_match:
            signal['stop_loss'] = float(sl_match.group(1))
            signal['sl_pips'] = float(sl_match.group(2))

        tp_match = re.search(r'Take Profit:\s*([\d.]+)\s*\(([\d.]+)\s*pips\)', prev_block)
        if tp_match:
            signal['take_profit'] = float(tp_match.group(1))
            signal['tp_pips'] = float(tp_match.group(2))

        # Extract timestamp from signal summary (look in prev block)
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', prev_block[-500:])
        if ts_match:
            signal['timestamp'] = ts_match.group(1)

        signals.append(signal)

    return signals

def extract_outcomes(log_file):
    """Extract signal outcomes from end of log"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Find the summary section
    summary_match = re.search(r'📊 RESULTS BY EPIC:(.*?)✅ TOTAL SMC_STRUCTURE SIGNALS', content, re.DOTALL)

    # Extract wins/losses from final stats
    winners = []
    losers = []

    # Look for winner/loser information
    # Since we don't have explicit outcome in log, we'll infer from ACT_P and ACT_L

    stats_match = re.search(r'✅ Winners:\s*(\d+).*?❌ Losers:\s*(\d+)', content, re.DOTALL)
    if stats_match:
        total_winners = int(stats_match.group(1))
        total_losers = int(stats_match.group(2))

        return {
            'total_winners': total_winners,
            'total_losers': total_losers,
            'win_rate': total_winners / (total_winners + total_losers) * 100
        }

    return None

def main():
    log_file = '/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/logs/smc_15m_v2.5.0_validation_20251111.log'

    print("Extracting signals from log...")
    signals = extract_signals(log_file)

    print(f"\n Total signals extracted: {len(signals)}")

    # Analyze by zone
    zone_counts = defaultdict(int)
    zone_with_validation = defaultdict(int)

    for sig in signals:
        zone = sig.get('entry_zone', 'UNKNOWN')
        zone_counts[zone] += 1

        if 'zone_validation' in sig:
            zone_with_validation[zone] += 1

    print("\nZone Distribution:")
    for zone, count in sorted(zone_counts.items()):
        print(f"  {zone}: {count} signals")

    print("\nZone Validation Captured:")
    for zone, count in sorted(zone_with_validation.items()):
        print(f"  {zone}: {count}/{zone_counts[zone]} have validation data")

    # Export to JSON
    output_file = '/home/hr/Projects/TradeSystemV1/smc_signals_extracted.json'
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"\nSignals exported to: {output_file}")

    # Extract outcomes
    outcomes = extract_outcomes(log_file)
    if outcomes:
        print(f"\nOutcome Summary:")
        print(f"  Winners: {outcomes['total_winners']}")
        print(f"  Losers: {outcomes['total_losers']}")
        print(f"  Win Rate: {outcomes['win_rate']:.1f}%")

    # Sample signal
    if signals:
        print("\nSample Signal (first one):")
        print(json.dumps(signals[0], indent=2))

if __name__ == '__main__':
    main()
