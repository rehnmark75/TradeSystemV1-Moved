#!/usr/bin/env python3
"""
Analyze SMC Strategy v2.5.0 Entry Positions
Purpose: Identify if we're taking bull entries near swing highs and bear entries near swing lows
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

LOG_FILE = "/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/logs/smc_15m_v2.5.0_validation_20251111.log"

def parse_signals(log_file: str) -> List[Dict]:
    """Parse all signals from the log file with their entry position data."""
    signals = []
    current_signal = {}
    in_signal_block = False

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for premium/discount validation section
            if "STEP 3D: Premium/Discount Zone Entry Timing Validation" in line:
                in_signal_block = True
                current_signal = {}
                continue

            if in_signal_block:
                # Extract range data
                if "Range Analysis" in line:
                    match = re.search(r'(\d+\.\d+) pips', line)
                    if match:
                        current_signal['range_pips'] = float(match.group(1))

                # Extract high/mid/low
                if "High:" in line and "High:" not in current_signal:
                    match = re.search(r'High:\s+([\d.]+)', line)
                    if match:
                        current_signal['range_high'] = float(match.group(1))

                if "Mid:" in line and "Mid:" not in current_signal:
                    match = re.search(r'Mid:\s+([\d.]+)', line)
                    if match:
                        current_signal['range_mid'] = float(match.group(1))

                if "Low:" in line and "Low:" not in current_signal:
                    match = re.search(r'Low:\s+([\d.]+)', line)
                    if match:
                        current_signal['range_low'] = float(match.group(1))

                # Extract zone
                if "Current Zone:" in line:
                    if "PREMIUM" in line:
                        current_signal['zone'] = "PREMIUM"
                    elif "DISCOUNT" in line:
                        current_signal['zone'] = "DISCOUNT"
                    elif "EQUILIBRIUM" in line:
                        current_signal['zone'] = "EQUILIBRIUM"

                # Extract price position
                if "Price Position:" in line:
                    match = re.search(r'(\d+\.\d+)% of range', line)
                    if match:
                        current_signal['price_position_pct'] = float(match.group(1))

                # Extract HTF trend
                if "HTF Trend Context:" in line:
                    if "BULL" in line:
                        match = re.search(r'BULL.*strength:\s+(\d+)%', line)
                        if match:
                            current_signal['htf_trend'] = "BULL"
                            current_signal['htf_strength'] = int(match.group(1))
                    elif "BEAR" in line:
                        match = re.search(r'BEAR.*strength:\s+(\d+)%', line)
                        if match:
                            current_signal['htf_trend'] = "BEAR"
                            current_signal['htf_strength'] = int(match.group(1))

                # Check if signal passed or rejected
                if "BULLISH entry in" in line:
                    current_signal['direction'] = "BULL"
                    if "excellent timing" in line or "TREND CONTINUATION" in line:
                        current_signal['passed'] = True
                    elif "poor timing" in line:
                        current_signal['passed'] = False
                        current_signal['reject_reason'] = "premium_discount"

                if "BEARISH entry in" in line:
                    current_signal['direction'] = "BEAR"
                    if "excellent timing" in line or "TREND CONTINUATION" in line:
                        current_signal['passed'] = True
                    elif "poor timing" in line:
                        current_signal['passed'] = False
                        current_signal['reject_reason'] = "premium_discount"

                # Extract entry quality
                if "Entry quality:" in line:
                    match = re.search(r'Entry quality:\s+(\d+)%', line)
                    if match:
                        current_signal['entry_quality_pct'] = int(match.group(1))

                # End of signal block - look for signal summary
                if "VALID SMC STRUCTURE SIGNAL DETECTED" in line or "Rejected at premium/discount filter" in line:
                    in_signal_block = False
                    if current_signal and 'direction' in current_signal and 'price_position_pct' in current_signal:
                        signals.append(current_signal.copy())
                    current_signal = {}

            # Also capture actual executed signals with outcomes
            if "Direction:" in line and ("BULL" in line or "BEAR" in line):
                # This might be a signal summary - skip for now, we already captured from validation
                pass

    return signals

def analyze_entry_positions(signals: List[Dict]) -> None:
    """Analyze entry positions and identify problems."""

    print("="*80)
    print("SMC STRATEGY v2.5.0 - ENTRY POSITION ANALYSIS")
    print("="*80)
    print()

    # Filter to only passed signals
    passed_signals = [s for s in signals if s.get('passed') == True]
    rejected_signals = [s for s in signals if s.get('passed') == False]

    print(f"Total Signal Attempts: {len(signals)}")
    print(f"Passed Signals: {len(passed_signals)}")
    print(f"Rejected Signals: {len(rejected_signals)}")
    print()

    # Analyze BULL entries
    bull_passed = [s for s in passed_signals if s['direction'] == 'BULL']
    bear_passed = [s for s in passed_signals if s['direction'] == 'BEAR']

    print("="*80)
    print("BULL ENTRY ANALYSIS")
    print("="*80)
    print(f"Total BULL Signals: {len(bull_passed)}")
    print()

    # Position distribution for BULL entries
    bull_low = [s for s in bull_passed if s['price_position_pct'] < 33.33]
    bull_mid = [s for s in bull_passed if 33.33 <= s['price_position_pct'] <= 66.67]
    bull_high = [s for s in bull_passed if s['price_position_pct'] > 66.67]

    print(f"BULL Entries by Range Position:")
    print(f"  Lower 33% (DISCOUNT): {len(bull_low)} ({len(bull_low)/len(bull_passed)*100:.1f}%)")
    print(f"  Middle 33% (EQUILIBRIUM): {len(bull_mid)} ({len(bull_mid)/len(bull_passed)*100:.1f}%)")
    print(f"  Upper 33% (PREMIUM): {len(bull_high)} ({len(bull_high)/len(bull_passed)*100:.1f}%)")
    print()

    if bull_passed:
        avg_bull_position = sum(s['price_position_pct'] for s in bull_passed) / len(bull_passed)
        print(f"Average BULL Entry Position: {avg_bull_position:.1f}% of range")
        print()

    # Show examples of BULL entries in PREMIUM zone (bad positioning)
    if bull_high:
        print(f"WARNING: {len(bull_high)} BULL entries in PREMIUM zone (buying near highs)")
        print("\nExample BULL entries in PREMIUM zone:")
        for i, sig in enumerate(bull_high[:5], 1):
            print(f"  {i}. Position: {sig['price_position_pct']:.1f}% | Zone: {sig['zone']} | "
                  f"Entry Quality: {sig.get('entry_quality_pct', 'N/A')}%")
        print()

    print("="*80)
    print("BEAR ENTRY ANALYSIS")
    print("="*80)
    print(f"Total BEAR Signals: {len(bear_passed)}")
    print()

    # Position distribution for BEAR entries
    bear_low = [s for s in bear_passed if s['price_position_pct'] < 33.33]
    bear_mid = [s for s in bear_passed if 33.33 <= s['price_position_pct'] <= 66.67]
    bear_high = [s for s in bear_passed if s['price_position_pct'] > 66.67]

    print(f"BEAR Entries by Range Position:")
    print(f"  Lower 33% (DISCOUNT): {len(bear_low)} ({len(bear_low)/len(bear_passed)*100:.1f}%)")
    print(f"  Middle 33% (EQUILIBRIUM): {len(bear_mid)} ({len(bear_mid)/len(bear_passed)*100:.1f}%)")
    print(f"  Upper 33% (PREMIUM): {len(bear_high)} ({len(bear_high)/len(bear_passed)*100:.1f}%)")
    print()

    if bear_passed:
        avg_bear_position = sum(s['price_position_pct'] for s in bear_passed) / len(bear_passed)
        print(f"Average BEAR Entry Position: {avg_bear_position:.1f}% of range")
        print()

    # Show examples of BEAR entries in DISCOUNT zone (bad positioning)
    if bear_low:
        print(f"WARNING: {len(bear_low)} BEAR entries in DISCOUNT zone (selling near lows)")
        print("\nExample BEAR entries in DISCOUNT zone:")
        for i, sig in enumerate(bear_low[:5], 1):
            print(f"  {i}. Position: {sig['price_position_pct']:.1f}% | Zone: {sig['zone']} | "
                  f"HTF Trend: {sig['htf_trend']} ({sig['htf_strength']}%)")
        print()

    print("="*80)
    print("REJECTED SIGNALS ANALYSIS")
    print("="*80)

    bull_rejected = [s for s in rejected_signals if s['direction'] == 'BULL']
    bear_rejected = [s for s in rejected_signals if s['direction'] == 'BEAR']

    print(f"BULL Rejected: {len(bull_rejected)}")
    print(f"BEAR Rejected: {len(bear_rejected)}")
    print()

    # Show why BULL signals were rejected
    if bull_rejected:
        bull_rejected_premium = [s for s in bull_rejected if s.get('zone') == 'PREMIUM']
        print(f"BULL rejected due to PREMIUM zone: {len(bull_rejected_premium)}")
        if bull_rejected_premium:
            avg_rejected_position = sum(s['price_position_pct'] for s in bull_rejected_premium) / len(bull_rejected_premium)
            print(f"  Average position: {avg_rejected_position:.1f}% of range")

    # Show why BEAR signals were rejected
    if bear_rejected:
        bear_rejected_discount = [s for s in bear_rejected if s.get('zone') == 'DISCOUNT']
        print(f"BEAR rejected due to DISCOUNT zone: {len(bear_rejected_discount)}")
        if bear_rejected_discount:
            avg_rejected_position = sum(s['price_position_pct'] for s in bear_rejected_discount) / len(bear_rejected_discount)
            print(f"  Average position: {avg_rejected_position:.1f}% of range")

    print()
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    # Calculate problematic entries
    if bull_passed:
        bull_premium_pct = len(bull_high) / len(bull_passed) * 100
        print(f"1. {bull_premium_pct:.1f}% of BULL entries are in PREMIUM zone (buying near highs)")
        print(f"   This is PROBLEMATIC - we're buying after price has already rallied")
        print()

    if bear_passed:
        bear_discount_pct = len(bear_low) / len(bear_passed) * 100
        print(f"2. {bear_discount_pct:.1f}% of BEAR entries are in DISCOUNT zone (selling near lows)")
        print(f"   This is PROBLEMATIC - we're selling after price has already fallen")
        print()

    # Check HTF trend strength for wrong-zone entries
    print("3. HTF Trend Context Analysis:")
    print()

    # BEAR entries in DISCOUNT with weak HTF trend
    bear_low_weak_trend = [s for s in bear_low if s.get('htf_strength', 0) < 80]
    if bear_low_weak_trend:
        print(f"   BEAR entries in DISCOUNT with weak HTF trend (<80%): {len(bear_low_weak_trend)}")
        print(f"   These are likely counter-trend entries that will fail")
        print()

    print("="*80)
    print("RECOMMENDED FIXES")
    print("="*80)
    print()

    print("FIX #1: Tighten Premium/Discount Filter")
    print("-" * 40)
    print("Current Logic:")
    print("  - BULL allowed in DISCOUNT zone")
    print("  - BEAR allowed in DISCOUNT zone if HTF trend strength >= 80%")
    print()
    print("ISSUE: BEAR entries in DISCOUNT zone are problematic even with strong HTF trend")
    print("       We're selling near swing lows, which is poor risk/reward")
    print()
    print("RECOMMENDED CHANGE:")
    print("  - BULL: Only allow if price_position <= 33% (strict DISCOUNT)")
    print("  - BEAR: Only allow if price_position >= 67% (strict PREMIUM)")
    print("  - Remove HTF trend exception for BEAR in DISCOUNT")
    print()

    print("FIX #2: Add Distance from Swing Points")
    print("-" * 40)
    print("  - For BULL: Reject if entry is within 20% of recent swing HIGH")
    print("  - For BEAR: Reject if entry is within 20% of recent swing LOW")
    print("  - This ensures we have room to run before hitting resistance/support")
    print()

    print("FIX #3: Entry Quality Threshold")
    print("-" * 40)
    print("  - Current system calculates 'entry_quality_pct'")
    print("  - Add minimum threshold: entry_quality >= 50%")
    print("  - This filters out marginal entries")
    print()

if __name__ == "__main__":
    print("Parsing log file...")
    signals = parse_signals(LOG_FILE)
    print(f"Found {len(signals)} signals with position data\n")

    analyze_entry_positions(signals)
