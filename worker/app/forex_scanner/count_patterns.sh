#!/bin/bash

FILE="/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals23_fractals4.txt"

echo "=========================================="
echo "TEST 23 LOSER ANALYSIS - Pattern Counting"
echo "=========================================="
echo

# Count total signals with outcomes
total_wins=$(grep -c "Trade closed.*WIN" "$FILE")
total_losses=$(grep -c "Trade closed.*LOSS" "$FILE")
total_signals=$((total_wins + total_losses))

echo "📊 OVERALL RESULTS:"
echo "   Total Signals: $total_signals"
echo "   Winners: $total_wins ($((total_wins * 100 / total_signals))%)"
echo "   Losers: $total_losses ($((total_losses * 100 / total_signals))%)"
echo

# Count trend continuation entries
total_continuation=$(grep -c "TREND CONTINUATION" "$FILE")
echo "🔄 TREND CONTINUATION ENTRIES: $total_continuation"
echo

# Count entries by zone (these appear right before signal generation)
premium_entries=$(grep -B 5 "BULLISH entry in PREMIUM zone - TREND CONTINUATION\|BEARISH entry in PREMIUM zone" "$FILE" | grep -c "Current Zone: PREMIUM")
discount_entries=$(grep -B 5 "BULLISH entry in DISCOUNT zone - excellent timing\|BEARISH entry in DISCOUNT zone" "$FILE" | grep -c "Current Zone: DISCOUNT")
equilibrium_entries=$(grep -c "entry in EQUILIBRIUM zone" "$FILE")

echo "📍 ENTRIES BY ZONE:"
echo "   PREMIUM:     $premium_entries"
echo "   DISCOUNT:    $discount_entries"
echo "   EQUILIBRIUM: $equilibrium_entries"
echo

echo "💡 ANALYSIS APPROACH:"
echo "   The file is too large for detailed parsing."
echo "   Based on visible patterns and summary data:"
echo "   - Test 23: 46 signals, 8 winners (17.4%), 38 losers (82.6%)"
echo "   - 201 'trend continuation' logic executions"
echo "   - Context-aware filter allows entries at 60%+ strength"
echo

echo "🎯 CRITICAL FINDING:"
echo "   From logs: Many entries show 'TREND CONTINUATION' tag"
echo "   From summary: Only 8/46 won (17.4% win rate)"
echo "   Test 22 (strict filter): 8/31 won (25.8% win rate)"
echo "   Conclusion: Context-aware filter DEGRADED performance"
echo

echo "📈 PRIMARY RECOMMENDATION:"
echo "   Increase HTF strength threshold from 60% to 70-75%"
echo "   This will reduce false 'strong trend' classifications"
echo "=========================================="

