#!/usr/bin/env python3
"""
Analyze Test 23 losing trades to identify patterns
"""
import re
from collections import defaultdict

def extract_signals_from_file(filepath):
    """Extract all signal data with zones, HTF context, and outcomes"""

    signals = []
    current_signal = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for signal generation
        if '🎯 SMC_STRUCTURE SIGNAL GENERATED' in line:
            current_signal = {}

            # Look ahead for signal details
            for j in range(i, min(i + 200, len(lines))):
                ahead_line = lines[j]

                # Extract epic
                if 'Epic:' in ahead_line and 'epic' not in current_signal:
                    match = re.search(r'Epic:\s+(\w+)', ahead_line)
                    if match:
                        current_signal['epic'] = match.group(1)

                # Extract direction
                if 'Direction:' in ahead_line and 'direction' not in current_signal:
                    if 'BUY' in ahead_line:
                        current_signal['direction'] = 'BUY'
                    elif 'SELL' in ahead_line:
                        current_signal['direction'] = 'SELL'

                # Extract price
                if 'Entry Price:' in ahead_line and 'price' not in current_signal:
                    match = re.search(r'Entry Price:\s+([\d.]+)', ahead_line)
                    if match:
                        current_signal['price'] = float(match.group(1))

                # Extract zone
                if 'Current Zone:' in ahead_line and 'zone' not in current_signal:
                    if 'PREMIUM' in ahead_line:
                        current_signal['zone'] = 'PREMIUM'
                    elif 'DISCOUNT' in ahead_line:
                        current_signal['zone'] = 'DISCOUNT'
                    elif 'EQUILIBRIUM' in ahead_line:
                        current_signal['zone'] = 'EQUILIBRIUM'

                # Extract HTF trend info
                if 'HTF Trend:' in ahead_line and 'htf_trend' not in current_signal:
                    if 'BULL' in ahead_line:
                        current_signal['htf_trend'] = 'BULL'
                    elif 'BEAR' in ahead_line:
                        current_signal['htf_trend'] = 'BEAR'

                # Extract HTF strength
                if 'HTF Strength:' in ahead_line and 'htf_strength' not in current_signal:
                    match = re.search(r'(\d+)%', ahead_line)
                    if match:
                        current_signal['htf_strength'] = int(match.group(1))

                # Check if it's a trend continuation entry
                if 'TREND CONTINUATION' in ahead_line and 'is_continuation' not in current_signal:
                    current_signal['is_continuation'] = True

                # Look for trade outcome
                if 'Trade closed' in ahead_line:
                    if 'LOSS' in ahead_line:
                        current_signal['outcome'] = 'LOSS'
                        # Extract loss amount
                        match = re.search(r'-?([\d.]+)\s+pips', ahead_line)
                        if match:
                            current_signal['pips'] = -float(match.group(1))
                    elif 'WIN' in ahead_line:
                        current_signal['outcome'] = 'WIN'
                        # Extract win amount
                        match = re.search(r'\+?([\d.]+)\s+pips', ahead_line)
                        if match:
                            current_signal['pips'] = float(match.group(1))

                    # Signal complete, add it
                    if 'outcome' in current_signal and current_signal.get('zone'):
                        if 'is_continuation' not in current_signal:
                            current_signal['is_continuation'] = False
                        signals.append(current_signal.copy())
                    break

        i += 1

    return signals


def analyze_losers(signals):
    """Analyze losing trades for patterns"""

    losers = [s for s in signals if s.get('outcome') == 'LOSS']
    winners = [s for s in signals if s.get('outcome') == 'WIN']

    print(f"\n📊 EXTRACTED {len(signals)} SIGNALS: {len(winners)} winners, {len(losers)} losers")
    print("=" * 80)

    # Analyze losers by zone
    print("\n📍 LOSERS BY ZONE:")
    print("-" * 80)
    losers_by_zone = defaultdict(list)
    for s in losers:
        zone = s.get('zone', 'UNKNOWN')
        losers_by_zone[zone].append(s)

    for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT', 'UNKNOWN']:
        count = len(losers_by_zone[zone])
        pct = (count / len(losers) * 100) if losers else 0
        print(f"   {zone:12} {count:3} losers ({pct:5.1f}%)")

    # Analyze trend continuation losers
    print("\n🔄 TREND CONTINUATION ANALYSIS:")
    print("-" * 80)
    continuation_losers = [s for s in losers if s.get('is_continuation')]
    pullback_losers = [s for s in losers if not s.get('is_continuation')]

    print(f"   Trend Continuation Losers: {len(continuation_losers)} ({len(continuation_losers)/len(losers)*100:.1f}%)")
    print(f"   Pullback Losers:          {len(pullback_losers)} ({len(pullback_losers)/len(losers)*100:.1f}%)")

    # Analyze by HTF strength
    print("\n💪 LOSERS BY HTF STRENGTH:")
    print("-" * 80)
    losers_by_strength = defaultdict(list)
    for s in losers:
        strength = s.get('htf_strength')
        if strength:
            if strength >= 80:
                bucket = '80-100%'
            elif strength >= 70:
                bucket = '70-79%'
            elif strength >= 60:
                bucket = '60-69%'
            else:
                bucket = '<60%'
            losers_by_strength[bucket].append(s)

    for bucket in ['80-100%', '70-79%', '60-69%', '<60%']:
        count = len(losers_by_strength[bucket])
        pct = (count / len(losers) * 100) if losers else 0
        print(f"   {bucket:10} {count:3} losers ({pct:5.1f}%)")

    # Critical pattern: Zone + Strength + Continuation
    print("\n🎯 CRITICAL PATTERN ANALYSIS:")
    print("-" * 80)

    # Count specific problematic patterns
    patterns = {
        'BULL + PREMIUM + 60% + CONTINUATION': 0,
        'BULL + PREMIUM + 70%+ + CONTINUATION': 0,
        'BEAR + DISCOUNT + 60% + CONTINUATION': 0,
        'BULL + DISCOUNT (pullback)': 0,
        'BEAR + PREMIUM (pullback)': 0,
        'EQUILIBRIUM (any)': 0,
    }

    for s in losers:
        zone = s.get('zone', '')
        direction = s.get('direction', '')
        htf_trend = s.get('htf_trend', '')
        strength = s.get('htf_strength', 0)
        is_cont = s.get('is_continuation', False)

        if direction == 'BUY' and zone == 'PREMIUM' and strength >= 60 and strength < 70 and is_cont:
            patterns['BULL + PREMIUM + 60% + CONTINUATION'] += 1
        elif direction == 'BUY' and zone == 'PREMIUM' and strength >= 70 and is_cont:
            patterns['BULL + PREMIUM + 70%+ + CONTINUATION'] += 1
        elif direction == 'SELL' and zone == 'DISCOUNT' and strength >= 60 and is_cont:
            patterns['BEAR + DISCOUNT + 60% + CONTINUATION'] += 1
        elif direction == 'BUY' and zone == 'DISCOUNT' and not is_cont:
            patterns['BULL + DISCOUNT (pullback)'] += 1
        elif direction == 'SELL' and zone == 'PREMIUM' and not is_cont:
            patterns['BEAR + PREMIUM (pullback)'] += 1
        elif zone == 'EQUILIBRIUM':
            patterns['EQUILIBRIUM (any)'] += 1

    for pattern, count in patterns.items():
        pct = (count / len(losers) * 100) if losers else 0
        print(f"   {pattern:40} {count:3} ({pct:5.1f}%)")

    # Compare to winners
    print("\n✅ WINNERS FOR COMPARISON:")
    print("-" * 80)

    winners_by_zone = defaultdict(list)
    for s in winners:
        zone = s.get('zone', 'UNKNOWN')
        winners_by_zone[zone].append(s)

    for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT', 'UNKNOWN']:
        count = len(winners_by_zone[zone])
        pct = (count / len(winners) * 100) if winners else 0
        print(f"   {zone:12} {count:3} winners ({pct:5.1f}%)")

    continuation_winners = [s for s in winners if s.get('is_continuation')]
    pullback_winners = [s for s in winners if not s.get('is_continuation')]

    print(f"\n   Trend Continuation Winners: {len(continuation_winners)} ({len(continuation_winners)/len(winners)*100 if winners else 0:.1f}%)")
    print(f"   Pullback Winners:          {len(pullback_winners)} ({len(pullback_winners)/len(winners)*100 if winners else 0:.1f}%)")

    # Show detailed loser examples
    print("\n📋 SAMPLE LOSING TRADES:")
    print("-" * 80)
    for i, s in enumerate(losers[:10]):
        print(f"\nLoser #{i+1}:")
        print(f"   Epic: {s.get('epic', 'N/A')}")
        print(f"   Direction: {s.get('direction', 'N/A')}")
        print(f"   Zone: {s.get('zone', 'N/A')}")
        print(f"   HTF Trend: {s.get('htf_trend', 'N/A')} @ {s.get('htf_strength', 'N/A')}%")
        print(f"   Type: {'CONTINUATION' if s.get('is_continuation') else 'PULLBACK'}")
        print(f"   Loss: {s.get('pips', 'N/A')} pips")

    print("\n" + "=" * 80)

    # RECOMMENDATION
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 80)

    bull_premium_60_pct = (patterns['BULL + PREMIUM + 60% + CONTINUATION'] / len(losers) * 100) if losers else 0

    if bull_premium_60_pct > 20:
        print("   ⚠️  HIGH RISK PATTERN: BULL + PREMIUM + 60% strength")
        print("   📈 Recommendation: Increase strength threshold from 60% to 70-75%")

    if len(losers_by_zone['EQUILIBRIUM']) > len(losers_by_zone['DISCOUNT']):
        print("   ⚠️  Equilibrium zone has more losers than discount zone")
        print("   📈 Recommendation: Apply stricter filtering to equilibrium entries")

    continuation_loss_rate = (len(continuation_losers) / (len(continuation_losers) + len(continuation_winners)) * 100) if (len(continuation_losers) + len(continuation_winners)) > 0 else 0
    pullback_loss_rate = (len(pullback_losers) / (len(pullback_losers) + len(pullback_winners)) * 100) if (len(pullback_losers) + len(pullback_winners)) > 0 else 0

    print(f"\n   Continuation Loss Rate: {continuation_loss_rate:.1f}%")
    print(f"   Pullback Loss Rate: {pullback_loss_rate:.1f}%")

    if continuation_loss_rate > pullback_loss_rate + 10:
        print("   ⚠️  Trend continuation entries performing WORSE than pullbacks")
        print("   📈 Recommendation: Re-evaluate context-aware filter logic")


if __name__ == '__main__':
    filepath = '/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals23_fractals4.txt'
    signals = extract_signals_from_file(filepath)
    analyze_losers(signals)
