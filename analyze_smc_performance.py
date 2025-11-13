#!/usr/bin/env python3
"""
Analyze SMC strategy performance without needing trade outcomes
Focus on signal characteristics and zone performance
"""

import json
import statistics
from collections import defaultdict

def load_signals(file_path):
    """Load extracted signals"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_by_zone(signals):
    """Analyze signals by entry zone"""
    print("\n" + "="*80)
    print("ZONE-BASED ANALYSIS")
    print("="*80)

    zones = defaultdict(list)
    for sig in signals:
        zone = sig.get('entry_zone', 'UNKNOWN')
        zones[zone].append(sig)

    # Print zone distribution
    print("\n📊 ZONE DISTRIBUTION:")
    print(f"{'Zone':<15} {'Count':<10} {'% of Total':<15}")
    print("-" * 40)
    for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT', 'UNKNOWN']:
        count = len(zones[zone])
        pct = count / len(signals) * 100
        print(f"{zone:<15} {count:<10} {pct:.1f}%")

    # Analyze characteristics by zone
    print("\n📊 ZONE CHARACTERISTICS:")
    for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT']:
        if not zones[zone]:
            continue

        print(f"\n--- {zone} ZONE ---")
        zone_sigs = zones[zone]

        # HTF strength
        htf_strengths = [s['htf_strength'] for s in zone_sigs if 'htf_strength' in s]
        if htf_strengths:
            print(f"HTF Strength: Mean={statistics.mean(htf_strengths):.1f}%, Median={statistics.median(htf_strengths):.1f}%")

        # R:R ratio
        rr_ratios = [s['rr_ratio'] for s in zone_sigs if 'rr_ratio' in s]
        if rr_ratios:
            print(f"R:R Ratio: Mean={statistics.mean(rr_ratios):.2f}, Median={statistics.median(rr_ratios):.2f}")

        # Price position
        positions = [s['price_position'] for s in zone_sigs if 'price_position' in s]
        if positions:
            print(f"Price Position: Mean={statistics.mean(positions):.1f}%, Median={statistics.median(positions):.1f}%")

        # Direction breakdown
        bull_count = len([s for s in zone_sigs if s.get('direction') == 'BULL'])
        bear_count = len([s for s in zone_sigs if s.get('direction') == 'BEAR'])
        print(f"Direction: BULL={bull_count}, BEAR={bear_count}")

        # HTF alignment
        aligned = len([s for s in zone_sigs if s.get('direction') == s.get('htf_trend')])
        counter = len(zone_sigs) - aligned
        print(f"HTF Alignment: Aligned={aligned}, Counter-trend={counter}")

        # Zone validation
        passed = len([s for s in zone_sigs if s.get('zone_validation') == 'PASS'])
        failed = len([s for s in zone_sigs if s.get('zone_validation') == 'FAIL'])
        print(f"Zone Validation: PASS={passed}, FAIL={failed}")

    return zones

def analyze_equilibrium_deep_dive(signals):
    """Deep analysis of equilibrium zone"""
    print("\n" + "="*80)
    print("EQUILIBRIUM ZONE DEEP DIVE")
    print("="*80)

    eq_signals = [s for s in signals if s.get('entry_zone') == 'EQUILIBRIUM']

    if not eq_signals:
        print("No equilibrium signals found!")
        return

    print(f"\nTotal Equilibrium Signals: {len(eq_signals)}")

    # User reported 15.4% WR for equilibrium
    # With 10 signals, that's ~1-2 winners
    estimated_winners = round(len(eq_signals) * 0.154)
    print(f"Estimated Winners (15.4% WR): ~{estimated_winners}")
    print(f"Estimated Losers: ~{len(eq_signals) - estimated_winners}")

    # Analyze what makes equilibrium different
    print("\n--- Equilibrium vs Other Zones ---")

    non_eq = [s for s in signals if s.get('entry_zone') in ['PREMIUM', 'DISCOUNT']]

    metrics = {
        'htf_strength': 'HTF Strength',
        'rr_ratio': 'R:R Ratio',
        'price_position': 'Price Position'
    }

    for metric_key, metric_name in metrics.items():
        eq_values = [s[metric_key] for s in eq_signals if metric_key in s]
        non_eq_values = [s[metric_key] for s in non_eq if metric_key in s]

        if eq_values and non_eq_values:
            eq_mean = statistics.mean(eq_values)
            non_eq_mean = statistics.mean(non_eq_values)
            diff = eq_mean - non_eq_mean

            print(f"{metric_name}:")
            print(f"  Equilibrium: {eq_mean:.2f}")
            print(f"  Premium/Discount: {non_eq_mean:.2f}")
            print(f"  Difference: {diff:+.2f}")

    # HTF trend context
    print("\n--- Equilibrium HTF Trend Context ---")
    htf_trends = defaultdict(int)
    for sig in eq_signals:
        trend = sig.get('htf_trend', 'UNKNOWN')
        htf_trends[trend] += 1

    for trend, count in sorted(htf_trends.items()):
        print(f"  {trend}: {count} signals")

    # Direction vs HTF alignment
    aligned_eq = len([s for s in eq_signals if s.get('direction') == s.get('htf_trend')])
    counter_eq = len(eq_signals) - aligned_eq
    print(f"\nHTF Alignment:")
    print(f"  Aligned: {aligned_eq} ({aligned_eq/len(eq_signals)*100:.1f}%)")
    print(f"  Counter-trend: {counter_eq} ({counter_eq/len(eq_signals)*100:.1f}%)")

    # List all equilibrium signals
    print("\n--- All Equilibrium Signals ---")
    print(f"{'#':<4} {'Pair':<8} {'Dir':<5} {'HTF Trend':<10} {'HTF%':<6} {'PricePos':<10} {'R:R':<6} {'Validation':<12}")
    print("-" * 90)
    for sig in eq_signals:
        sig_id = sig['signal_id']
        pair = sig.get('pair', 'N/A')
        direction = sig.get('direction', 'N/A')
        htf_trend = sig.get('htf_trend', 'N/A')
        htf_str = f"{sig.get('htf_strength', 0):.0f}"
        pos = f"{sig.get('price_position', 0):.1f}"
        rr = f"{sig.get('rr_ratio', 0):.2f}"
        val = sig.get('zone_validation', 'N/A')

        print(f"{sig_id:<4} {pair:<8} {direction:<5} {htf_trend:<10} {htf_str:<6} {pos:<10} {rr:<6} {val:<12}")

def analyze_bull_vs_bear(signals):
    """Compare BULL vs BEAR performance"""
    print("\n" + "="*80)
    print("BULL VS BEAR ANALYSIS")
    print("="*80)

    bull_signals = [s for s in signals if s.get('direction') == 'BULL']
    bear_signals = [s for s in signals if s.get('direction') == 'BEAR']

    print(f"\nBULL Signals: {len(bull_signals)}")
    print(f"BEAR Signals: {len(bear_signals)}")

    # User reported: BULL ~30% WR, BEAR ~47.8% WR
    bull_estimated_wr = 30.0
    bear_estimated_wr = 47.8

    bull_winners = round(len(bull_signals) * bull_estimated_wr / 100)
    bear_winners = round(len(bear_signals) * bear_estimated_wr / 100)

    print(f"\nEstimated Performance:")
    print(f"BULL: ~{bull_winners} winners / {len(bull_signals)-bull_winners} losers ({bull_estimated_wr:.1f}% WR)")
    print(f"BEAR: ~{bear_winners} winners / {len(bear_signals)-bear_winners} losers ({bear_estimated_wr:.1f}% WR)")

    # Compare characteristics
    print("\n--- Metric Comparison ---")
    print(f"{'Metric':<25} {'BULL':<15} {'BEAR':<15} {'Difference':<15}")
    print("-" * 70)

    metrics = {
        'htf_strength': 'HTF Strength',
        'rr_ratio': 'R:R Ratio',
        'price_position': 'Price Position'
    }

    for metric_key, metric_name in metrics.items():
        bull_values = [s[metric_key] for s in bull_signals if metric_key in s]
        bear_values = [s[metric_key] for s in bear_signals if metric_key in s]

        if bull_values and bear_values:
            bull_mean = statistics.mean(bull_values)
            bear_mean = statistics.mean(bear_values)
            diff = bear_mean - bull_mean

            print(f"{metric_name:<25} {bull_mean:<15.2f} {bear_mean:<15.2f} {diff:+.2f}")

    # Zone distribution by direction
    print("\n--- Zone Distribution ---")
    print(f"{'Zone':<15} {'BULL':<15} {'BEAR':<15}")
    print("-" * 45)

    for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT']:
        bull_zone = len([s for s in bull_signals if s.get('entry_zone') == zone])
        bear_zone = len([s for s in bear_signals if s.get('entry_zone') == zone])
        print(f"{zone:<15} {bull_zone:<15} {bear_zone:<15}")

    # HTF alignment
    bull_aligned = len([s for s in bull_signals if s.get('htf_trend') == 'BULL'])
    bull_counter = len(bull_signals) - bull_aligned
    bear_aligned = len([s for s in bear_signals if s.get('htf_trend') == 'BEAR'])
    bear_counter = len(bear_signals) - bear_aligned

    print("\n--- HTF Alignment ---")
    print(f"BULL signals with BULL HTF: {bull_aligned} ({bull_aligned/len(bull_signals)*100:.1f}%)")
    print(f"BULL signals with BEAR/RANGING HTF: {bull_counter} ({bull_counter/len(bull_signals)*100:.1f}%)")
    print(f"BEAR signals with BEAR HTF: {bear_aligned} ({bear_aligned/len(bear_signals)*100:.1f}%)")
    print(f"BEAR signals with BULL/RANGING HTF: {bear_counter} ({bear_counter/len(bear_signals)*100:.1f}%)")

def analyze_zone_validation(signals):
    """Analyze zone validation pass/fail impact"""
    print("\n" + "="*80)
    print("ZONE VALIDATION ANALYSIS")
    print("="*80)

    passed = [s for s in signals if s.get('zone_validation') == 'PASS']
    failed = [s for s in signals if s.get('zone_validation') == 'FAIL']

    print(f"\nZone Validation PASS: {len(passed)}")
    print(f"Zone Validation FAIL: {len(failed)}")

    # Analyze what causes failures
    print("\n--- Validation Failures by Zone ---")
    fail_by_zone = defaultdict(int)
    for sig in failed:
        zone = sig.get('entry_zone', 'UNKNOWN')
        fail_by_zone[zone] += 1

    for zone, count in sorted(fail_by_zone.items()):
        total_zone = len([s for s in signals if s.get('entry_zone') == zone])
        pct = count / total_zone * 100 if total_zone > 0 else 0
        print(f"  {zone}: {count}/{total_zone} ({pct:.1f}% fail rate)")

    # HTF strength comparison
    if passed and failed:
        pass_htf = [s['htf_strength'] for s in passed if 'htf_strength' in s]
        fail_htf = [s['htf_strength'] for s in failed if 'htf_strength' in s]

        if pass_htf and fail_htf:
            print(f"\nHTF Strength:")
            print(f"  PASS: Mean={statistics.mean(pass_htf):.1f}%")
            print(f"  FAIL: Mean={statistics.mean(fail_htf):.1f}%")

def generate_recommendations(signals):
    """Generate data-driven recommendations"""
    print("\n" + "="*80)
    print("ACTIONABLE RECOMMENDATIONS")
    print("="*80)

    # 1. Equilibrium zone filter
    eq_signals = [s for s in signals if s.get('entry_zone') == 'EQUILIBRIUM']
    eq_wr = 15.4  # User reported

    print("\n1. EQUILIBRIUM ZONE FILTER")
    print(f"   Current: Equilibrium accepted (50% confidence threshold)")
    print(f"   Performance: {eq_wr:.1f}% WR ({len(eq_signals)} signals)")
    print(f"   RECOMMENDATION: EXCLUDE equilibrium zone OR raise confidence to 70%+")
    print(f"   Expected Impact:")
    print(f"      - Remove {len(eq_signals)} signals")
    print(f"      - New signal count: {len(signals) - len(eq_signals)}")

    # Calculate new WR without equilibrium
    total_winners = 22  # User reported
    eq_winners = round(len(eq_signals) * 0.154)
    remaining_winners = total_winners - eq_winners
    remaining_signals = len(signals) - len(eq_signals)
    new_wr = remaining_winners / remaining_signals * 100 if remaining_signals > 0 else 0

    print(f"      - New WR: {new_wr:.1f}% (vs 31.0% current)")
    print(f"      - WR Improvement: +{new_wr - 31.0:.1f}%")

    # 2. HTF strength threshold
    print("\n2. HTF STRENGTH OPTIMIZATION")

    # Check current HTF distribution
    htf_strengths = [s['htf_strength'] for s in signals if 'htf_strength' in s]
    if htf_strengths:
        print(f"   Current minimum: 75% (but many signals at 60%)")
        print(f"   Mean HTF strength: {statistics.mean(htf_strengths):.1f}%")

        # Count signals at different thresholds
        at_85_plus = len([s for s in signals if s.get('htf_strength', 0) >= 85])
        at_80_plus = len([s for s in signals if s.get('htf_strength', 0) >= 80])
        at_75_plus = len([s for s in signals if s.get('htf_strength', 0) >= 75])

        print(f"   Signals at 85%+: {at_85_plus}")
        print(f"   Signals at 80%+: {at_80_plus}")
        print(f"   Signals at 75%+: {at_75_plus}")

        if at_85_plus < len(signals) * 0.3:
            print(f"   RECOMMENDATION: Consider 80% threshold (keeps {at_80_plus} signals)")
        else:
            print(f"   RECOMMENDATION: Raise to 85% threshold (keeps {at_85_plus} signals)")

    # 3. Zone-specific strategies
    print("\n3. ZONE-SPECIFIC STRATEGY")
    print(f"   Premium Zone: 45.8% WR - BEST PERFORMER")
    print(f"   Discount Zone: 16.7% WR - POOR PERFORMER")
    print(f"   Equilibrium Zone: 15.4% WR - WORST PERFORMER")
    print(f"   ")
    print(f"   RECOMMENDATION: Focus on PREMIUM zone entries only")

    premium_signals = [s for s in signals if s.get('entry_zone') == 'PREMIUM']
    premium_winners = round(len(premium_signals) * 0.458)

    print(f"   Expected Performance:")
    print(f"      - Signals: {len(premium_signals)}")
    print(f"      - Winners: ~{premium_winners}")
    print(f"      - WR: 45.8%")
    print(f"      - WR Improvement: +{45.8 - 31.0:.1f}%")

    # 4. Directional bias
    print("\n4. DIRECTIONAL BIAS")
    bull_signals = [s for s in signals if s.get('direction') == 'BULL']
    bear_signals = [s for s in signals if s.get('direction') == 'BEAR']

    print(f"   BULL: 30% WR ({len(bull_signals)} signals)")
    print(f"   BEAR: 47.8% WR ({len(bear_signals)} signals)")
    print(f"   ")
    print(f"   RECOMMENDATION: Apply stricter filters to BULL signals")
    print(f"      - Require HTF alignment (BULL HTF for BULL signals)")
    print(f"      - Require premium zone entries for BULL")
    print(f"      - Or: Focus primarily on BEAR signals")

    # 5. Combined optimal filter
    print("\n5. COMBINED OPTIMAL FILTER (Recommended)")
    print(f"   Filter combination:")
    print(f"      - Entry Zone: PREMIUM only")
    print(f"      - Direction: BEAR preferred (or BULL with HTF alignment)")
    print(f"      - HTF Strength: 80%+ minimum")

    # Count signals meeting this criteria
    optimal_signals = [
        s for s in signals
        if s.get('entry_zone') == 'PREMIUM'
        and s.get('htf_strength', 0) >= 80
    ]

    bear_optimal = [s for s in optimal_signals if s.get('direction') == 'BEAR']

    print(f"   ")
    print(f"   Expected Results:")
    print(f"      - Premium + 80% HTF: {len(optimal_signals)} signals")
    print(f"      - Premium + 80% HTF + BEAR: {len(bear_optimal)} signals")
    print(f"      - Estimated WR: 50-55% (premium zone advantage + strong HTF)")

def main():
    # Load signals
    signals = load_signals('/home/hr/Projects/TradeSystemV1/smc_signals_extracted.json')

    print("="*80)
    print("SMC_STRUCTURE STRATEGY - DEEP PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nTotal Signals: {len(signals)}")
    print(f"Win Rate: 31.0% (22 winners / 49 losers)")
    print(f"Profit Factor: 0.52")

    # Run all analyses
    analyze_by_zone(signals)
    analyze_equilibrium_deep_dive(signals)
    analyze_bull_vs_bear(signals)
    analyze_zone_validation(signals)
    generate_recommendations(signals)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
