#!/usr/bin/env python3
"""
Deep analysis of SMC_STRUCTURE strategy performance
Focus: Equilibrium zone failures and winner vs loser characteristics
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any
import statistics

class SMCAnalysis:
    def __init__(self, log_file_path: str):
        self.log_file = log_file_path
        self.signals = []

    def parse_log(self):
        """Parse backtest log and extract all signals with full context"""
        print(f"Parsing {self.log_file}...")

        current_signal = {}
        in_signal_block = False

        with open(self.log_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Detect signal start
                if 'SMC_STRUCTURE signal detected' in line or 'Signal detected' in line:
                    if current_signal:
                        self.signals.append(current_signal)
                    current_signal = {}
                    in_signal_block = True

                    # Extract timestamp
                    ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if ts_match:
                        current_signal['timestamp'] = ts_match.group(1)

                if not in_signal_block:
                    continue

                # Extract key fields
                if 'Pair:' in line:
                    match = re.search(r'Pair:\s*(\w+)', line)
                    if match:
                        current_signal['pair'] = match.group(1)

                # Extract zone from validation step
                if 'Current Zone:' in line:
                    match = re.search(r'Current Zone:\s*(PREMIUM|DISCOUNT|EQUILIBRIUM)', line)
                    if match:
                        current_signal['entry_zone'] = match.group(1)

                # Extract price position for zone analysis
                if 'Price Position:' in line:
                    match = re.search(r'Price Position:\s*([\d.]+)%', line)
                    if match:
                        current_signal['price_position'] = float(match.group(1))

                # Extract zone entry result
                if 'entry in' in line and 'zone' in line:
                    if 'BULLISH entry in' in line or 'BEARISH entry in' in line:
                        # Extract pass/fail
                        if '✅' in line:
                            current_signal['zone_validation'] = 'PASS'
                            if 'TREND CONTINUATION' in line:
                                current_signal['zone_reason'] = 'TREND_CONTINUATION'
                            elif 'excellent timing' in line:
                                current_signal['zone_reason'] = 'EXCELLENT_TIMING'
                        elif '❌' in line:
                            current_signal['zone_validation'] = 'FAIL'
                            if 'poor timing' in line:
                                current_signal['zone_reason'] = 'POOR_TIMING'

                if 'Direction:' in line:
                    match = re.search(r'Direction:\s*(BULL|BEAR)', line)
                    if match:
                        current_signal['direction'] = match.group(1)

                if 'Entry TF:' in line:
                    match = re.search(r'Entry TF:\s*(\w+)', line)
                    if match:
                        current_signal['entry_tf'] = match.group(1)

                if 'HTF:' in line:
                    match = re.search(r'HTF:\s*(\w+)', line)
                    if match:
                        current_signal['htf'] = match.group(1)

                if 'Price:' in line:
                    match = re.search(r'Price:\s*([\d.]+)', line)
                    if match:
                        current_signal['entry_price'] = float(match.group(1))

                if 'HTF Strength:' in line:
                    match = re.search(r'HTF Strength:\s*([\d.]+)', line)
                    if match:
                        current_signal['htf_strength'] = float(match.group(1))

                if 'Entry Zone:' in line:
                    match = re.search(r'Entry Zone:\s*(\w+)', line)
                    if match:
                        current_signal['entry_zone'] = match.group(1).upper()

                if 'Zone Confidence:' in line:
                    match = re.search(r'Zone Confidence:\s*([\d.]+)%', line)
                    if match:
                        current_signal['zone_confidence'] = float(match.group(1))

                if 'RR Ratio:' in line or 'R:R Ratio:' in line:
                    match = re.search(r'(?:RR|R:R) Ratio:\s*([\d.]+)', line)
                    if match:
                        current_signal['rr_ratio'] = float(match.group(1))

                if 'Stop Loss:' in line:
                    match = re.search(r'Stop Loss:\s*([\d.]+)', line)
                    if match:
                        current_signal['stop_loss'] = float(match.group(1))

                if 'Take Profit:' in line:
                    match = re.search(r'Take Profit:\s*([\d.]+)', line)
                    if match:
                        current_signal['take_profit'] = float(match.group(1))

                # Extract BOS/CHoCH info
                if 'BOS confirmed' in line:
                    current_signal['structure_type'] = 'BOS'
                elif 'CHoCH confirmed' in line:
                    current_signal['structure_type'] = 'CHoCH'

                if 'Structure Confidence:' in line:
                    match = re.search(r'Structure Confidence:\s*([\d.]+)%', line)
                    if match:
                        current_signal['structure_confidence'] = float(match.group(1))

                # Extract outcome
                if 'WINNER' in line:
                    current_signal['outcome'] = 'WIN'
                    match = re.search(r'profit:\s*\+?([\d.]+)%', line)
                    if match:
                        current_signal['profit_pct'] = float(match.group(1))
                elif 'LOSER' in line:
                    current_signal['outcome'] = 'LOSS'
                    match = re.search(r'loss:\s*-?([\d.]+)%', line)
                    if match:
                        current_signal['profit_pct'] = -float(match.group(1))

                # Detect end of signal block
                if 'Result:' in line and ('WINNER' in line or 'LOSER' in line):
                    in_signal_block = False

        # Don't forget last signal
        if current_signal:
            self.signals.append(current_signal)

        print(f"Parsed {len(self.signals)} signals")
        return self.signals

    def analyze_equilibrium_zone(self):
        """Deep dive into equilibrium zone failures"""
        print("\n" + "="*80)
        print("EQUILIBRIUM ZONE DEEP DIVE")
        print("="*80)

        eq_signals = [s for s in self.signals if s.get('entry_zone') == 'EQUILIBRIUM']
        eq_winners = [s for s in eq_signals if s.get('outcome') == 'WIN']
        eq_losers = [s for s in eq_signals if s.get('outcome') == 'LOSS']

        print(f"\nTotal Equilibrium Signals: {len(eq_signals)}")
        print(f"Winners: {len(eq_winners)}")
        print(f"Losers: {len(eq_losers)}")
        print(f"Win Rate: {len(eq_winners)/len(eq_signals)*100:.1f}%")

        # Compare to other zones
        premium_signals = [s for s in self.signals if s.get('entry_zone') == 'PREMIUM']
        discount_signals = [s for s in self.signals if s.get('entry_zone') == 'DISCOUNT']

        premium_wr = len([s for s in premium_signals if s.get('outcome') == 'WIN']) / len(premium_signals) * 100 if premium_signals else 0
        discount_wr = len([s for s in discount_signals if s.get('outcome') == 'WIN']) / len(discount_signals) * 100 if discount_signals else 0

        print(f"\nComparison:")
        print(f"Premium Zone WR: {premium_wr:.1f}% ({len(premium_signals)} signals)")
        print(f"Discount Zone WR: {discount_wr:.1f}% ({len(discount_signals)} signals)")
        print(f"Equilibrium Zone WR: {len(eq_winners)/len(eq_signals)*100:.1f}% ({len(eq_signals)} signals)")

        # Analyze equilibrium characteristics
        if eq_signals:
            print("\n--- Equilibrium Zone Characteristics ---")

            # HTF Strength
            eq_htf = [s['htf_strength'] for s in eq_signals if 'htf_strength' in s]
            if eq_htf:
                print(f"HTF Strength - Mean: {statistics.mean(eq_htf):.1f}%, Median: {statistics.median(eq_htf):.1f}%")

            # Zone Confidence
            eq_conf = [s['zone_confidence'] for s in eq_signals if 'zone_confidence' in s]
            if eq_conf:
                print(f"Zone Confidence - Mean: {statistics.mean(eq_conf):.1f}%, Median: {statistics.median(eq_conf):.1f}%")

            # R:R Ratio
            eq_rr = [s['rr_ratio'] for s in eq_signals if 'rr_ratio' in s]
            if eq_rr:
                print(f"R:R Ratio - Mean: {statistics.mean(eq_rr):.2f}, Median: {statistics.median(eq_rr):.2f}")

            # Direction breakdown
            eq_bull = len([s for s in eq_signals if s.get('direction') == 'BULL'])
            eq_bear = len([s for s in eq_signals if s.get('direction') == 'BEAR'])
            eq_bull_wr = len([s for s in eq_signals if s.get('direction') == 'BULL' and s.get('outcome') == 'WIN']) / eq_bull * 100 if eq_bull else 0
            eq_bear_wr = len([s for s in eq_signals if s.get('direction') == 'BEAR' and s.get('outcome') == 'WIN']) / eq_bear * 100 if eq_bear else 0

            print(f"\nDirection Breakdown:")
            print(f"BULL: {eq_bull} signals, {eq_bull_wr:.1f}% WR")
            print(f"BEAR: {eq_bear} signals, {eq_bear_wr:.1f}% WR")

            # Structure type
            eq_bos = len([s for s in eq_signals if s.get('structure_type') == 'BOS'])
            eq_choch = len([s for s in eq_signals if s.get('structure_type') == 'CHoCH'])
            print(f"\nStructure Type:")
            print(f"BOS: {eq_bos} signals")
            print(f"CHoCH: {eq_choch} signals")

        return eq_signals, eq_winners, eq_losers

    def analyze_winners_vs_losers(self):
        """Compare winners vs losers across ALL zones"""
        print("\n" + "="*80)
        print("WINNERS VS LOSERS - COMPREHENSIVE ANALYSIS")
        print("="*80)

        winners = [s for s in self.signals if s.get('outcome') == 'WIN']
        losers = [s for s in self.signals if s.get('outcome') == 'LOSS']

        print(f"\nTotal Signals: {len(self.signals)}")
        print(f"Winners: {len(winners)} ({len(winners)/len(self.signals)*100:.1f}%)")
        print(f"Losers: {len(losers)} ({len(losers)/len(self.signals)*100:.1f}%)")

        metrics = {
            'htf_strength': 'HTF Strength',
            'zone_confidence': 'Zone Confidence',
            'structure_confidence': 'Structure Confidence',
            'rr_ratio': 'R:R Ratio'
        }

        print("\n--- Key Metrics Comparison ---")
        print(f"{'Metric':<25} {'Winners':<15} {'Losers':<15} {'Difference':<15}")
        print("-" * 70)

        for metric_key, metric_name in metrics.items():
            winner_values = [s[metric_key] for s in winners if metric_key in s]
            loser_values = [s[metric_key] for s in losers if metric_key in s]

            if winner_values and loser_values:
                winner_mean = statistics.mean(winner_values)
                loser_mean = statistics.mean(loser_values)
                diff = winner_mean - loser_mean
                diff_pct = (diff / loser_mean * 100) if loser_mean != 0 else 0

                print(f"{metric_name:<25} {winner_mean:<15.2f} {loser_mean:<15.2f} {diff:+.2f} ({diff_pct:+.1f}%)")

        # Zone distribution
        print("\n--- Zone Distribution ---")
        for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT']:
            zone_winners = len([s for s in winners if s.get('entry_zone') == zone])
            zone_losers = len([s for s in losers if s.get('entry_zone') == zone])
            zone_total = zone_winners + zone_losers
            zone_wr = zone_winners / zone_total * 100 if zone_total else 0
            print(f"{zone:<15} {zone_winners:>3}W / {zone_losers:>3}L / {zone_total:>3} Total = {zone_wr:.1f}% WR")

        # Direction distribution
        print("\n--- Direction Distribution ---")
        bull_winners = len([s for s in winners if s.get('direction') == 'BULL'])
        bull_losers = len([s for s in losers if s.get('direction') == 'BULL'])
        bear_winners = len([s for s in winners if s.get('direction') == 'BEAR'])
        bear_losers = len([s for s in losers if s.get('direction') == 'BEAR'])

        bull_total = bull_winners + bull_losers
        bear_total = bear_winners + bear_losers

        bull_wr = bull_winners / bull_total * 100 if bull_total else 0
        bear_wr = bear_winners / bear_total * 100 if bear_total else 0

        print(f"BULL: {bull_winners}W / {bull_losers}L / {bull_total} Total = {bull_wr:.1f}% WR")
        print(f"BEAR: {bear_winners}W / {bear_losers}L / {bear_total} Total = {bear_wr:.1f}% WR")

        # Structure type
        print("\n--- Structure Type Distribution ---")
        bos_winners = len([s for s in winners if s.get('structure_type') == 'BOS'])
        bos_losers = len([s for s in losers if s.get('structure_type') == 'BOS'])
        choch_winners = len([s for s in winners if s.get('structure_type') == 'CHoCH'])
        choch_losers = len([s for s in losers if s.get('structure_type') == 'CHoCH'])

        bos_total = bos_winners + bos_losers
        choch_total = choch_winners + choch_losers

        bos_wr = bos_winners / bos_total * 100 if bos_total else 0
        choch_wr = choch_winners / choch_total * 100 if choch_total else 0

        print(f"BOS: {bos_winners}W / {bos_losers}L / {bos_total} Total = {bos_wr:.1f}% WR")
        print(f"CHoCH: {choch_winners}W / {choch_losers}L / {choch_total} Total = {choch_wr:.1f}% WR")

        return winners, losers

    def analyze_bull_vs_bear(self):
        """Analyze what makes BEAR signals better than BULL"""
        print("\n" + "="*80)
        print("BULL VS BEAR PERFORMANCE ANALYSIS")
        print("="*80)

        bull_signals = [s for s in self.signals if s.get('direction') == 'BULL']
        bear_signals = [s for s in self.signals if s.get('direction') == 'BEAR']

        bull_winners = [s for s in bull_signals if s.get('outcome') == 'WIN']
        bear_winners = [s for s in bear_signals if s.get('outcome') == 'WIN']

        bull_wr = len(bull_winners) / len(bull_signals) * 100 if bull_signals else 0
        bear_wr = len(bear_winners) / len(bear_signals) * 100 if bear_signals else 0

        print(f"\nBULL Signals: {len(bull_signals)}, WR: {bull_wr:.1f}%")
        print(f"BEAR Signals: {len(bear_signals)}, WR: {bear_wr:.1f}%")
        print(f"Difference: {bear_wr - bull_wr:+.1f}% (BEAR advantage)")

        # Compare metrics
        metrics = {
            'htf_strength': 'HTF Strength',
            'zone_confidence': 'Zone Confidence',
            'structure_confidence': 'Structure Confidence',
            'rr_ratio': 'R:R Ratio'
        }

        print("\n--- Metric Comparison ---")
        print(f"{'Metric':<25} {'BULL':<15} {'BEAR':<15} {'Difference':<15}")
        print("-" * 70)

        for metric_key, metric_name in metrics.items():
            bull_values = [s[metric_key] for s in bull_signals if metric_key in s]
            bear_values = [s[metric_key] for s in bear_signals if metric_key in s]

            if bull_values and bear_values:
                bull_mean = statistics.mean(bull_values)
                bear_mean = statistics.mean(bear_values)
                diff = bear_mean - bull_mean

                print(f"{metric_name:<25} {bull_mean:<15.2f} {bear_mean:<15.2f} {diff:+.2f}")

        # Zone distribution
        print("\n--- Zone Distribution ---")
        print(f"{'Zone':<15} {'BULL WR':<15} {'BEAR WR':<15} {'Difference':<15}")
        print("-" * 60)

        for zone in ['PREMIUM', 'EQUILIBRIUM', 'DISCOUNT']:
            bull_zone = [s for s in bull_signals if s.get('entry_zone') == zone]
            bear_zone = [s for s in bear_signals if s.get('entry_zone') == zone]

            bull_zone_wr = len([s for s in bull_zone if s.get('outcome') == 'WIN']) / len(bull_zone) * 100 if bull_zone else 0
            bear_zone_wr = len([s for s in bear_zone if s.get('outcome') == 'WIN']) / len(bear_zone) * 100 if bear_zone else 0

            print(f"{zone:<15} {bull_zone_wr:<15.1f} {bear_zone_wr:<15.1f} {bear_zone_wr - bull_zone_wr:+.1f}")

        return bull_signals, bear_signals

    def generate_recommendations(self):
        """Generate data-driven recommendations for filters"""
        print("\n" + "="*80)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*80)

        # Analyze current thresholds
        winners = [s for s in self.signals if s.get('outcome') == 'WIN']
        losers = [s for s in self.signals if s.get('outcome') == 'LOSS']

        # 1. Equilibrium zone filter
        eq_signals = [s for s in self.signals if s.get('entry_zone') == 'EQUILIBRIUM']
        eq_wr = len([s for s in eq_signals if s.get('outcome') == 'WIN']) / len(eq_signals) * 100 if eq_signals else 0

        print("\n1. EQUILIBRIUM ZONE FILTER")
        print(f"   Current: Equilibrium zone accepted with 50% confidence threshold")
        print(f"   Performance: {eq_wr:.1f}% WR ({len(eq_signals)} signals)")

        if eq_wr < 35:
            print(f"   RECOMMENDATION: EXCLUDE equilibrium zone entries OR raise confidence to 70%+")
            print(f"   Expected Impact: Remove {len(eq_signals)} low-performing signals")
            print(f"   Expected WR Improvement: +{(len(winners)-len([s for s in winners if s.get('entry_zone') == 'EQUILIBRIUM']))/(len(self.signals)-len(eq_signals))*100 - len(winners)/len(self.signals)*100:.1f}%")

        # 2. HTF strength threshold
        winner_htf = [s['htf_strength'] for s in winners if 'htf_strength' in s]
        loser_htf = [s['htf_strength'] for s in losers if 'htf_strength' in s]

        if winner_htf and loser_htf:
            winner_htf_mean = statistics.mean(winner_htf)
            loser_htf_mean = statistics.mean(loser_htf)

            print("\n2. HTF STRENGTH THRESHOLD")
            print(f"   Current: 75% minimum")
            print(f"   Winners average: {winner_htf_mean:.1f}%")
            print(f"   Losers average: {loser_htf_mean:.1f}%")

            if winner_htf_mean > loser_htf_mean + 5:
                recommended_threshold = 85
                filtered_signals = len([s for s in self.signals if s.get('htf_strength', 0) >= recommended_threshold])
                filtered_winners = len([s for s in winners if s.get('htf_strength', 0) >= recommended_threshold])

                print(f"   RECOMMENDATION: Raise threshold to {recommended_threshold}%")
                print(f"   Expected Impact: {filtered_signals} signals remaining")
                print(f"   Expected WR: {filtered_winners/filtered_signals*100:.1f}%")

        # 3. R:R ratio threshold
        winner_rr = [s['rr_ratio'] for s in winners if 'rr_ratio' in s]
        loser_rr = [s['rr_ratio'] for s in losers if 'rr_ratio' in s]

        if winner_rr and loser_rr:
            winner_rr_mean = statistics.mean(winner_rr)
            loser_rr_mean = statistics.mean(loser_rr)

            print("\n3. R:R RATIO THRESHOLD")
            print(f"   Current: 1.2:1 minimum")
            print(f"   Winners average: {winner_rr_mean:.2f}:1")
            print(f"   Losers average: {loser_rr_mean:.2f}:1")

            if winner_rr_mean > loser_rr_mean + 0.1:
                recommended_threshold = 1.5
                filtered_signals = len([s for s in self.signals if s.get('rr_ratio', 0) >= recommended_threshold])
                filtered_winners = len([s for s in winners if s.get('rr_ratio', 0) >= recommended_threshold])

                print(f"   RECOMMENDATION: Raise threshold to {recommended_threshold}:1")
                print(f"   Expected Impact: {filtered_signals} signals remaining")
                print(f"   Expected WR: {filtered_winners/filtered_signals*100:.1f}%")

        # 4. Zone-specific confidence thresholds
        print("\n4. ZONE-SPECIFIC CONFIDENCE THRESHOLDS")
        for zone in ['PREMIUM', 'DISCOUNT', 'EQUILIBRIUM']:
            zone_signals = [s for s in self.signals if s.get('entry_zone') == zone]
            zone_winners = [s for s in zone_signals if s.get('outcome') == 'WIN']
            zone_wr = len(zone_winners) / len(zone_signals) * 100 if zone_signals else 0

            zone_conf = [s['zone_confidence'] for s in zone_signals if 'zone_confidence' in s]
            if zone_conf:
                mean_conf = statistics.mean(zone_conf)
                print(f"   {zone}: {zone_wr:.1f}% WR, avg confidence: {mean_conf:.1f}%")

        # 5. Directional bias
        bull_wr = len([s for s in self.signals if s.get('direction') == 'BULL' and s.get('outcome') == 'WIN']) / len([s for s in self.signals if s.get('direction') == 'BULL']) * 100
        bear_wr = len([s for s in self.signals if s.get('direction') == 'BEAR' and s.get('outcome') == 'WIN']) / len([s for s in self.signals if s.get('direction') == 'BEAR']) * 100

        print("\n5. DIRECTIONAL PERFORMANCE")
        print(f"   BULL: {bull_wr:.1f}% WR")
        print(f"   BEAR: {bear_wr:.1f}% WR")

        if abs(bull_wr - bear_wr) > 15:
            print(f"   RECOMMENDATION: Consider directional weighting or stricter filters for underperforming direction")

    def export_detailed_csv(self, output_file='smc_detailed_analysis.csv'):
        """Export all signals to CSV for external analysis"""
        import csv

        fieldnames = ['timestamp', 'pair', 'direction', 'entry_zone', 'entry_tf', 'htf',
                     'entry_price', 'stop_loss', 'take_profit', 'htf_strength',
                     'zone_confidence', 'structure_confidence', 'structure_type',
                     'rr_ratio', 'outcome', 'profit_pct']

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for signal in self.signals:
                writer.writerow(signal)

        print(f"\nDetailed signal data exported to: {output_file}")


def main():
    # Analyze v2.5.0
    print("="*80)
    print("SMC_STRUCTURE DEEP PERFORMANCE ANALYSIS - v2.5.0")
    print("="*80)

    log_file = '/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/logs/smc_15m_v2.5.0_validation_20251111.log'

    analyzer = SMCAnalysis(log_file)
    analyzer.parse_log()

    # Run all analyses
    analyzer.analyze_equilibrium_zone()
    analyzer.analyze_winners_vs_losers()
    analyzer.analyze_bull_vs_bear()
    analyzer.generate_recommendations()

    # Export CSV for further analysis
    analyzer.export_detailed_csv('/home/hr/Projects/TradeSystemV1/smc_detailed_analysis.csv')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
