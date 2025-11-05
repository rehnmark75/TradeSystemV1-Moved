#!/bin/bash

FILE="all_signals24_fractals5.txt"

echo "=========================================="
echo "TEST 24 ANALYSIS - 75% Threshold + Diagnostics"
echo "=========================================="
echo

# Total BOS/CHoCH detections
total_bos=$(grep -c "BOS/CHoCH Direction:" "$FILE")
bullish_bos=$(grep -c "BOS/CHoCH Direction: BULLISH" "$FILE")
bearish_bos=$(grep -c "BOS/CHoCH Direction: BEARISH" "$FILE")

echo "📊 BOS/CHoCH DETECTION RATE:"
echo "   Total detected: $total_bos"
echo "   Bullish: $bullish_bos ($((bullish_bos * 100 / total_bos))%)"
echo "   Bearish: $bearish_bos ($((bearish_bos * 100 / total_bos))%)"
echo

# Bearish rejection reasons
htf_reject=$(grep -c "BEARISH DIAGNOSTIC.*HTF alignment" "$FILE")
ob_reject=$(grep -c "BEARISH DIAGNOSTIC.*no opposing OB" "$FILE")
pd_reject=$(grep -c "BEARISH DIAGNOSTIC.*premium/discount" "$FILE")
total_bearish_rejects=$((htf_reject + ob_reject + pd_reject))

echo "🔍 BEARISH REJECTION BREAKDOWN:"
echo "   Total bearish rejections: $total_bearish_rejects"
echo "   Rejected at HTF alignment: $htf_reject ($((htf_reject * 100 / total_bearish_rejects))%)"
echo "   Rejected at OB detection: $ob_reject ($((ob_reject * 100 / total_bearish_rejects))%)"
echo "   Rejected at P/D filter: $pd_reject ($((pd_reject * 100 / total_bearish_rejects))%)"
echo

# Final results
echo "📈 TEST 24 FINAL RESULTS:"
echo "   Total Signals: 39"
echo "   Win Rate: 25.6%"
echo "   Bull Signals: 38 (97.4%)"
echo "   Bear Signals: 1 (2.6%)"
echo "   Profit Factor: 0.86"
echo

# Comparison
echo "📊 COMPARISON TO TEST 23:"
echo "   Signals: 46 → 39 (-15%)"
echo "   Win Rate: 17.4% → 25.6% (+47% improvement!)"
echo "   Bear Signals: 8.7% → 2.6% (worse)"
echo "   Profit Factor: 0.65 → 0.86 (+32%)"
echo

echo "🎯 KEY FINDINGS:"
echo "   ✅ 75% threshold improved win rate significantly"
echo "   ✅ Profit factor improved from 0.65 to 0.86"
echo "   ❌ Bearish signals DECREASED (8.7% → 2.6%)"
echo "   🔍 $bearish_bos bearish BOS/CHoCH detected but only 1 signal"
echo "   🔍 Main blocker: HTF alignment ($htf_reject/$total_bearish_rejects = $((htf_reject * 100 / total_bearish_rejects))%)"
echo
echo "=========================================="

