#!/bin/bash

FILE="/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/all_signals25_fractals6.txt"

echo "=========================================="
echo "TEST 25 DETAILED ANALYSIS - 60 Days"
echo "=========================================="
echo

# BOS/CHoCH detections
total_bos=$(grep -c "BOS/CHoCH Direction:" "$FILE")
bullish_bos=$(grep -c "BOS/CHoCH Direction: BULLISH" "$FILE")
bearish_bos=$(grep -c "BOS/CHoCH Direction: BEARISH" "$FILE")

echo "📊 BOS/CHoCH DETECTION (60 days):"
echo "   Total: $total_bos"
echo "   Bullish: $bullish_bos ($((bullish_bos * 100 / total_bos))%)"
echo "   Bearish: $bearish_bos ($((bearish_bos * 100 / total_bos))%)"
echo

# Bearish rejections
htf_reject=$(grep -c "BEARISH DIAGNOSTIC.*HTF alignment" "$FILE")
ob_reject=$(grep -c "BEARISH DIAGNOSTIC.*no opposing OB" "$FILE")
pd_reject=$(grep -c "BEARISH DIAGNOSTIC.*premium/discount" "$FILE")
total_bear_reject=$((htf_reject + ob_reject + pd_reject))

echo "🔍 BEARISH REJECTION BREAKDOWN (60 days):"
echo "   Total rejections: $total_bear_reject"
echo "   HTF alignment: $htf_reject ($((htf_reject * 100 / total_bear_reject))%)"
echo "   No OB found: $ob_reject ($((ob_reject * 100 / total_bear_reject))%)"
echo "   P/D filter: $pd_reject ($((pd_reject * 100 / total_bear_reject))%)"
echo

# Compare to Test 24 (30 days)
echo "📊 COMPARISON TO TEST 24 (30 days):"
echo ""
echo "   BOS/CHoCH Detections:"
echo "      Test 24 (30d): 704 total, 125 bearish (18%)"
echo "      Test 25 (60d): $total_bos total, $bearish_bos bearish ($((bearish_bos * 100 / total_bos))%)"
echo "      Rate: Test 24 = 23.5/day, Test 25 = $((total_bos / 60))/day"
echo ""
echo "   Bearish Rejections:"
echo "      Test 24 (30d): 137 total, 109 HTF (79%)"
echo "      Test 25 (60d): $total_bear_reject total, $htf_reject HTF ($((htf_reject * 100 / total_bear_reject))%)"
echo ""
echo "   Final Signals:"
echo "      Test 24 (30d): 39 signals (1.30/day)"
echo "      Test 25 (60d): 38 signals (0.63/day)"
echo ""

echo "⚠️  CRITICAL INSIGHT:"
if [ $total_bos -lt 1000 ]; then
    echo "   BOS/CHoCH detection rate DECREASED in 60-day period"
    echo "   This suggests the second 30 days had:"
    echo "   - Less trending behavior (fewer structure breaks)"
    echo "   - More ranging/choppy market conditions"
    echo "   - HTF alignment correctly filtered low-quality setups"
fi
echo

echo "📈 WIN RATE ANALYSIS:"
echo "   Test 24 (30d): 25.6% WR, 0.86 PF"
echo "   Test 25 (60d): 21.1% WR, 0.81 PF"
echo "   Degradation: -17% WR, -6% PF"
echo ""
echo "   Possible causes:"
echo "   1. Market regime change (trending → ranging)"
echo "   2. Lower quality setups in second 30 days"
echo "   3. Strategy performs better in strong trends"
echo

echo "=========================================="

