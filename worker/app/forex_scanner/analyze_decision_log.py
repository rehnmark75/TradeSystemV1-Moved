#!/usr/bin/env python3
"""
Analyze signal decision logs from backtest execution.
Provides comprehensive breakdown of rejection reasons and filter performance.
"""

import pandas as pd
import sys
from pathlib import Path
from collections import Counter

def analyze_decision_log(csv_path: str):
    """Analyze signal decision CSV and generate comprehensive report."""

    # Read CSV
    df = pd.read_csv(csv_path)

    print("=" * 100)
    print(" " * 30 + "SIGNAL DECISION ANALYSIS REPORT")
    print("=" * 100)
    print(f"Execution Log: {csv_path}")
    print(f"Total Evaluations: {len(df):,}")
    print()

    # Overall decision breakdown
    decision_counts = df['final_decision'].value_counts()
    approved = decision_counts.get('APPROVED', 0)
    rejected = decision_counts.get('REJECTED', 0)

    print("=" * 100)
    print("OVERALL DECISION BREAKDOWN")
    print("=" * 100)
    print(f"✅ Approved:  {approved:4d} ({approved/len(df)*100:5.2f}%)")
    print(f"❌ Rejected:  {rejected:4d} ({rejected/len(df)*100:5.2f}%)")
    print()

    # Rejection reasons breakdown
    if rejected > 0:
        print("=" * 100)
        print("REJECTION REASONS (Top 10)")
        print("=" * 100)
        rejected_df = df[df['final_decision'] == 'REJECTED']
        rejection_reasons = rejected_df['rejection_reason'].value_counts()

        for i, (reason, count) in enumerate(rejection_reasons.head(10).items(), 1):
            pct = count / rejected * 100
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"{i:2d}. {reason:30s} {count:5d} ({pct:5.1f}%) {bar}")
        print()

        # Rejection by step
        print("=" * 100)
        print("REJECTION BY FILTER STAGE")
        print("=" * 100)
        rejection_steps = rejected_df['rejection_step'].value_counts()

        for step, count in rejection_steps.items():
            pct = count / rejected * 100
            print(f"{step:30s} {count:5d} ({pct:5.1f}%)")
        print()

    # Direction breakdown (approved signals)
    if approved > 0:
        print("=" * 100)
        print("APPROVED SIGNALS - DIRECTION BREAKDOWN")
        print("=" * 100)
        approved_df = df[df['final_decision'] == 'APPROVED']
        direction_counts = approved_df['direction'].value_counts()

        for direction, count in direction_counts.items():
            pct = count / approved * 100
            print(f"{direction.capitalize():10s} {count:4d} ({pct:5.1f}%)")
        print()

        # Pair breakdown for approved signals
        print("=" * 100)
        print("APPROVED SIGNALS - BY CURRENCY PAIR")
        print("=" * 100)
        pair_counts = approved_df['pair'].value_counts()

        for pair, count in pair_counts.items():
            pct = count / approved * 100
            print(f"{pair:10s} {count:4d} ({pct:5.1f}%)")
        print()

    # HTF Trend Analysis
    print("=" * 100)
    print("HTF TREND ANALYSIS")
    print("=" * 100)
    htf_trend_counts = df['htf_trend'].value_counts()

    for trend, count in htf_trend_counts.items():
        pct = count / len(df) * 100
        print(f"{str(trend):10s} {count:5d} ({pct:5.1f}%)")
    print()

    # Premium/Discount Zone Analysis
    print("=" * 100)
    print("PREMIUM/DISCOUNT ZONE ANALYSIS")
    print("=" * 100)

    # Filter out empty zone values
    zone_df = df[df['premium_discount_zone'].notna() & (df['premium_discount_zone'] != '')]

    if len(zone_df) > 0:
        zone_counts = zone_df['premium_discount_zone'].value_counts()

        print("Zone Distribution:")
        for zone, count in zone_counts.items():
            pct = count / len(zone_df) * 100
            print(f"  {zone.capitalize():15s} {count:5d} ({pct:5.1f}%)")
        print()

        # Premium/Discount rejection analysis
        pd_rejected = df[df['rejection_reason'] == 'PREMIUM_DISCOUNT_REJECT']

        if len(pd_rejected) > 0:
            print(f"Premium/Discount Rejections: {len(pd_rejected)}")
            print()
            print("Breakdown by Direction + Zone:")

            for direction in ['bullish', 'bearish']:
                dir_df = pd_rejected[pd_rejected['direction'] == direction]
                if len(dir_df) > 0:
                    zones = dir_df['premium_discount_zone'].value_counts()
                    print(f"  {direction.capitalize()}:")
                    for zone, count in zones.items():
                        print(f"    {zone.capitalize():15s} {count:5d}")
    print()

    # Confidence Score Analysis (for signals that reached confidence check)
    print("=" * 100)
    print("CONFIDENCE SCORE ANALYSIS")
    print("=" * 100)

    # Filter signals with confidence scores
    conf_df = df[df['confidence'].notna() & (df['confidence'] != '')]

    if len(conf_df) > 0:
        # Convert confidence to float
        conf_df = conf_df.copy()
        conf_df['confidence'] = pd.to_numeric(conf_df['confidence'], errors='coerce')

        approved_conf = conf_df[conf_df['final_decision'] == 'APPROVED']['confidence']
        rejected_conf = conf_df[conf_df['final_decision'] == 'REJECTED']['confidence']

        print(f"Signals with Confidence Scores: {len(conf_df)}")
        print()

        if len(approved_conf) > 0:
            print(f"Approved Signals:")
            print(f"  Average: {approved_conf.mean():.3f}")
            print(f"  Min:     {approved_conf.min():.3f}")
            print(f"  Max:     {approved_conf.max():.3f}")
            print()

        if len(rejected_conf) > 0:
            print(f"Rejected Signals:")
            print(f"  Average: {rejected_conf.mean():.3f}")
            print(f"  Min:     {rejected_conf.min():.3f}")
            print(f"  Max:     {rejected_conf.max():.3f}")
    print()

    # Low confidence rejection details
    low_conf_rejected = df[df['rejection_reason'] == 'LOW_CONFIDENCE']

    if len(low_conf_rejected) > 0:
        print("=" * 100)
        print("LOW CONFIDENCE REJECTION DETAILS")
        print("=" * 100)
        print(f"Total Low Confidence Rejections: {len(low_conf_rejected)}")
        print()

        # Component score analysis
        lc_df = low_conf_rejected.copy()

        for col in ['confidence', 'htf_score', 'pattern_score', 'sr_score', 'rr_score']:
            if col in lc_df.columns:
                lc_df[col] = pd.to_numeric(lc_df[col], errors='coerce')
                values = lc_df[col].dropna()

                if len(values) > 0:
                    print(f"{col:20s} Avg: {values.mean():.3f}  Min: {values.min():.3f}  Max: {values.max():.3f}")
        print()

    # Sample approved signals
    if approved > 0:
        print("=" * 100)
        print("SAMPLE APPROVED SIGNALS (First 5)")
        print("=" * 100)

        approved_sample = approved_df.head(5)

        for idx, row in approved_sample.iterrows():
            print(f"Signal {idx + 1}:")
            print(f"  Pair:        {row['pair']}")
            print(f"  Direction:   {row['direction']}")
            print(f"  Timestamp:   {row['timestamp']}")
            print(f"  HTF Trend:   {row['htf_trend']} (strength: {row['htf_strength']})")
            print(f"  Zone:        {row['premium_discount_zone']}")
            print(f"  Confidence:  {row['confidence']}")
            print(f"  R:R Ratio:   {row['rr_ratio']}")
            print(f"  Entry:       {row['entry_price']}")
            print(f"  SL:          {row['stop_loss']}")
            print(f"  TP:          {row['take_profit']}")
            print()

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_decision_log.py <path_to_signal_decisions.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    analyze_decision_log(csv_path)
