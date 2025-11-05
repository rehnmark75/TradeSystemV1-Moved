#!/bin/bash

echo "=========================================="
echo "TEST 25 ANALYSIS - 60 Days vs Test 24 (30 Days)"
echo "=========================================="
echo

echo "📊 TEST 25 RESULTS (60 DAYS):"
echo "   Total Signals: 38"
echo "   Win Rate: 21.1%"
echo "   Winners: 8 (31.7 pips avg)"
echo "   Losers: 30 (10.4 pips avg)"
echo "   Profit Factor: 0.81"
echo "   Expectancy: -1.5 pips"
echo "   Bull Signals: 37 (97.4%)"
echo "   Bear Signals: 1 (2.6%)"
echo "   Avg Confidence: 43.8%"
echo

echo "📊 TEST 24 RESULTS (30 DAYS):"
echo "   Total Signals: 39"
echo "   Win Rate: 25.6%"
echo "   Winners: 10 (26.8 pips avg)"
echo "   Losers: 29 (10.7 pips avg)"
echo "   Profit Factor: 0.86"
echo "   Expectancy: -1.1 pips"
echo "   Bull Signals: 38 (97.4%)"
echo "   Bear Signals: 1 (2.6%)"
echo "   Avg Confidence: 43.6%"
echo

echo "⚠️  CRITICAL FINDINGS:"
echo "   60 days generated FEWER signals than 30 days!"
echo "   Signal rate: 38/60 = 0.63 signals/day"
echo "   Test 24 rate: 39/30 = 1.30 signals/day"
echo "   📉 Signal generation rate HALVED in second 30 days"
echo

echo "📉 PERFORMANCE DEGRADATION:"
echo "   Win Rate: 25.6% → 21.1% (-17% worse)"
echo "   Profit Factor: 0.86 → 0.81 (-6% worse)"
echo "   Expectancy: -1.1 → -1.5 (-36% worse)"
echo "   Avg Win: 26.8 → 31.7 pips (+18% better)"
echo

echo "🔍 HYPOTHESIS:"
echo "   The second 30 days (Sep 1 - Oct 1) may have been:"
echo "   1. Ranging/choppy market (HTF alignment rejects more)"
echo "   2. Bearish market (only 1 bear signal = HTF blocks them)"
echo "   3. Lower volatility (fewer BOS/CHoCH triggers)"
echo

echo "=========================================="

