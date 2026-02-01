#!/bin/bash
# Regime Filter A/B Test Script (v2 - Per-Pair Support)
# Tests global high volatility filter across multiple months
# Shows: Baseline, Per-Pair DB Setting, and Force Override
# Usage: ./regime_filter_test.sh [EPIC]
# Example: ./regime_filter_test.sh USDJPY

EPIC=${1:-USDJPY}
RESULTS_FILE="/tmp/regime_filter_results_${EPIC}_$(date +%Y%m%d_%H%M%S).txt"

echo "=========================================="
echo "REGIME FILTER A/B TEST - ${EPIC}"
echo "=========================================="
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Function to extract key metrics from backtest output
# Output format includes emojis: "📊 Total Signals: 48", "🎯 Win Rate: 25.0%", etc.
extract_metrics() {
    local output="$1"
    # Extract signals using awk (more reliable than sed with emoji lines)
    local signals=$(echo "$output" | grep "Total Signals:" | tail -1 | awk -F'Total Signals: ' '{print $2}' | awk '{print $1}')
    # Extract win rate
    local win_rate=$(echo "$output" | grep "Win Rate:" | tail -1 | awk -F'Win Rate: ' '{print $2}' | awk '{print $1}')
    # Extract profit factor
    local pf=$(echo "$output" | grep "Profit Factor:" | tail -1 | awk -F'Profit Factor: ' '{print $2}' | awk '{print $1}')
    # Extract expectancy (can be negative)
    local expectancy=$(echo "$output" | grep "Expectancy:" | tail -1 | awk -F'Expectancy: ' '{print $2}' | awk '{print $1}')
    echo "${signals:-0}|${win_rate:-0%}|${pf:-0}|${expectancy:-0}"
}

# Header for results
echo "REGIME FILTER A/B TEST RESULTS - ${EPIC}" > "$RESULTS_FILE"
echo "Generated: $(date)" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Filter Modes:" >> "$RESULTS_FILE"
echo "  OFF     = No historical intelligence (baseline)" >> "$RESULTS_FILE"
echo "  DB      = Uses per-pair DB setting (live behavior)" >> "$RESULTS_FILE"
echo "  FORCE   = Override to force filter ON" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
printf "%-12s | %-8s | %-8s | %-8s | %-6s | %-12s\n" "Period" "Filter" "Signals" "Win Rate" "PF" "Expectancy" >> "$RESULTS_FILE"
printf "%-12s-+-%-8s-+-%-8s-+-%-8s-+-%-6s-+-%-12s\n" "------------" "--------" "--------" "--------" "------" "------------" >> "$RESULTS_FILE"

# Test periods (Nov 2025, Dec 2025, Jan 2026)
declare -a PERIODS=(
    "2025-11-01|2025-11-30|Nov 2025"
    "2025-12-01|2025-12-31|Dec 2025"
    "2026-01-01|2026-01-30|Jan 2026"
)

for period in "${PERIODS[@]}"; do
    IFS='|' read -r START END LABEL <<< "$period"

    echo ""
    echo "Testing ${LABEL}..."

    # Test 1: Baseline (no intelligence, no filter)
    echo "  [1/3] Running baseline (no filter)..."
    OUTPUT_OFF=$(docker exec task-worker python /app/forex_scanner/bt.py ${EPIC} 30 \
        --scalp --timeframe 5m \
        --start-date ${START} --end-date ${END} 2>&1)

    METRICS_OFF=$(extract_metrics "$OUTPUT_OFF")
    IFS='|' read -r SIG_OFF WR_OFF PF_OFF EXP_OFF <<< "$METRICS_OFF"
    printf "%-12s | %-8s | %-8s | %-8s | %-6s | %-12s\n" "$LABEL" "OFF" "$SIG_OFF" "$WR_OFF" "$PF_OFF" "${EXP_OFF} pips" >> "$RESULTS_FILE"
    echo "       OFF: ${SIG_OFF} signals, ${WR_OFF} WR, ${PF_OFF} PF, ${EXP_OFF} pips"

    # Test 2: Per-pair DB setting (uses database scalp_block_global_high_volatility)
    echo "  [2/3] Running with DB per-pair setting..."
    OUTPUT_DB=$(docker exec task-worker python /app/forex_scanner/bt.py ${EPIC} 30 \
        --scalp --timeframe 5m \
        --start-date ${START} --end-date ${END} \
        --use-historical-intelligence 2>&1)

    METRICS_DB=$(extract_metrics "$OUTPUT_DB")
    IFS='|' read -r SIG_DB WR_DB PF_DB EXP_DB <<< "$METRICS_DB"
    printf "%-12s | %-8s | %-8s | %-8s | %-6s | %-12s\n" "$LABEL" "DB" "$SIG_DB" "$WR_DB" "$PF_DB" "${EXP_DB} pips" >> "$RESULTS_FILE"
    echo "        DB: ${SIG_DB} signals, ${WR_DB} WR, ${PF_DB} PF, ${EXP_DB} pips"

    # Test 3: Force filter ON (override)
    echo "  [3/3] Running with FORCE filter ON..."
    OUTPUT_FORCE=$(docker exec task-worker python /app/forex_scanner/bt.py ${EPIC} 30 \
        --scalp --timeframe 5m \
        --start-date ${START} --end-date ${END} \
        --use-historical-intelligence \
        --override scalp_block_global_high_volatility=true 2>&1)

    METRICS_FORCE=$(extract_metrics "$OUTPUT_FORCE")
    IFS='|' read -r SIG_FORCE WR_FORCE PF_FORCE EXP_FORCE <<< "$METRICS_FORCE"
    printf "%-12s | %-8s | %-8s | %-8s | %-6s | %-12s\n" "$LABEL" "FORCE" "$SIG_FORCE" "$WR_FORCE" "$PF_FORCE" "${EXP_FORCE} pips" >> "$RESULTS_FILE"
    echo "     FORCE: ${SIG_FORCE} signals, ${WR_FORCE} WR, ${PF_FORCE} PF, ${EXP_FORCE} pips"

    # Summary
    if [ "$SIG_OFF" != "0" ] && [ "$SIG_FORCE" != "0" ]; then
        BLOCKED=$((SIG_OFF - SIG_FORCE))
        BLOCKED_PCT=$((BLOCKED * 100 / SIG_OFF))
        echo "    → FORCE blocked ${BLOCKED} signals (${BLOCKED_PCT}%) vs baseline"
    fi
done

echo "" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "Notes:" >> "$RESULTS_FILE"
echo "  - OFF: No regime filter (baseline)" >> "$RESULTS_FILE"
echo "  - DB: Uses smc_simple_pair_overrides.scalp_block_global_high_volatility" >> "$RESULTS_FILE"
echo "  - FORCE: --override scalp_block_global_high_volatility=true" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "If DB == OFF, pair has filter disabled in database" >> "$RESULTS_FILE"
echo "If DB == FORCE, pair has filter enabled in database" >> "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results saved to: ${RESULTS_FILE}"
