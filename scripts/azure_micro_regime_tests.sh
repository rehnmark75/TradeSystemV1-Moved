#!/bin/bash
# =============================================================================
# Azure Micro-Regime Filter Tests
# Comprehensive testing of all micro-regime filters for scalp signal qualification
# =============================================================================
#
# PURPOSE: Run all micro-regime filter variations on Azure VM to find the best
#          filter combination for improving scalp trade win rate.
#
# BASELINE (from previous 180-day test):
#   - Signals: 6,091
#   - Win Rate: 56.9%
#   - Total Pips: +7,114.5
#
# EXPECTED: Individual filters should reduce signal count but improve win rate
#
# USAGE:
#   1. Push this script to Azure: scp -i ~/.ssh/azure_backtest_rsa scripts/azure_micro_regime_tests.sh azureuser@<IP>:/data/
#   2. SSH to Azure: ssh -i ~/.ssh/azure_backtest_rsa azureuser@<IP>
#   3. Run: chmod +x /data/azure_micro_regime_tests.sh && /data/azure_micro_regime_tests.sh
#
# Or run via azure_backtest.sh:
#   ./scripts/azure_backtest.sh ssh
#   Then: /data/azure_micro_regime_tests.sh
# =============================================================================

set -euo pipefail

# Configuration
DAYS=180
WORKERS=44
RESULTS_DIR="/data/micro_regime_results_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/test_log.txt"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Logging function
log() {
    echo -e "$1" | tee -a "${LOG_FILE}"
}

# Run a single test and capture results
run_test() {
    local test_name="$1"
    local test_args="$2"
    local output_file="${RESULTS_DIR}/${test_name}.txt"

    log ""
    log "${BLUE}============================================================${NC}"
    log "${BLUE}TEST: ${test_name}${NC}"
    log "${BLUE}Args: ${test_args}${NC}"
    log "${BLUE}Started: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    log "${BLUE}============================================================${NC}"

    # Run the backtest and capture output
    if docker exec task-worker python /app/forex_scanner/bt.py ${test_args} > "${output_file}" 2>&1; then
        log "${GREEN}✅ ${test_name} completed successfully${NC}"
    else
        log "${RED}❌ ${test_name} failed - check ${output_file}${NC}"
    fi

    # Extract key metrics from output
    log ""
    log "${YELLOW}Results Summary:${NC}"

    # Extract the summary line (contains signals, win rate, pips)
    if grep -E "Total.*signals|Win Rate|Total Pips|Profit Factor" "${output_file}" | tail -20 | tee -a "${LOG_FILE}"; then
        :
    else
        log "Could not extract summary metrics"
    fi

    # Also extract the final summary table if present
    if grep -A 30 "PARALLEL BACKTEST SUMMARY" "${output_file}" | head -35 | tee -a "${LOG_FILE}"; then
        :
    fi

    log ""
    log "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    log ""
}

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

log "${GREEN}============================================================${NC}"
log "${GREEN}MICRO-REGIME FILTER TESTING SUITE${NC}"
log "${GREEN}============================================================${NC}"
log "Started: $(date '+%Y-%m-%d %H:%M:%S')"
log "Days: ${DAYS}"
log "Workers: ${WORKERS}"
log "Results Dir: ${RESULTS_DIR}"
log ""

# Ensure task-worker is running
log "Checking task-worker container..."
if ! docker ps --format '{{.Names}}' | grep -q '^task-worker$'; then
    log "Starting task-worker container..."
    cd /data && docker compose up -d task-worker
    sleep 5
fi
log "${GREEN}✅ task-worker is running${NC}"
log ""

# =============================================================================
# TEST 1: BASELINE (No filters - for comparison)
# =============================================================================
run_test "01_baseline_no_filters" \
    "--all ${DAYS} --scalp --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 2: INDIVIDUAL MICRO-REGIME FILTERS
# =============================================================================

# 2a. Consecutive Candles Only (best local results: 61.5% WR vs 42.9% baseline)
run_test "02a_consecutive_candles_only" \
    "--all ${DAYS} --scalp --micro-consec-only --parallel --workers ${WORKERS}"

# 2b. Anti-Chop Only
run_test "02b_anti_chop_only" \
    "--all ${DAYS} --scalp --micro-antichop-only --parallel --workers ${WORKERS}"

# 2c. Body Dominance Only
run_test "02c_body_dominance_only" \
    "--all ${DAYS} --scalp --micro-body-only --parallel --workers ${WORKERS}"

# 2d. Micro-Range Only
run_test "02d_micro_range_only" \
    "--all ${DAYS} --scalp --micro-range-only --parallel --workers ${WORKERS}"

# 2e. Momentum Candle Only (stricter filter, disabled by default)
run_test "02e_momentum_candle_only" \
    "--all ${DAYS} --scalp --micro-momentum-only --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 3: ALL MICRO-REGIME FILTERS COMBINED
# =============================================================================
run_test "03_all_micro_regime_filters" \
    "--all ${DAYS} --scalp --micro-all-filters --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 4: MOMENTUM FILTERS ONLY (RSI, Two-Pole, MACD)
# =============================================================================
run_test "04_momentum_filters_only" \
    "--all ${DAYS} --scalp --micro-regime --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 5: BEST COMBINATIONS (based on expected results)
# =============================================================================

# 5a. Consecutive + Anti-Chop (expected strongest combo)
run_test "05a_consec_plus_antichop" \
    "--all ${DAYS} --scalp --micro-regime \
    --override scalp_consecutive_candles_enabled=true \
    --override scalp_anti_chop_enabled=true \
    --override scalp_body_dominance_enabled=false \
    --override scalp_micro_range_enabled=false \
    --override scalp_momentum_candle_enabled=false \
    --override scalp_rsi_filter_enabled=false \
    --override scalp_two_pole_filter_enabled=false \
    --override scalp_macd_filter_enabled=false \
    --parallel --workers ${WORKERS}"

# 5b. Consecutive + Body Dominance
run_test "05b_consec_plus_body" \
    "--all ${DAYS} --scalp --micro-regime \
    --override scalp_consecutive_candles_enabled=true \
    --override scalp_anti_chop_enabled=false \
    --override scalp_body_dominance_enabled=true \
    --override scalp_micro_range_enabled=false \
    --override scalp_momentum_candle_enabled=false \
    --override scalp_rsi_filter_enabled=false \
    --override scalp_two_pole_filter_enabled=false \
    --override scalp_macd_filter_enabled=false \
    --parallel --workers ${WORKERS}"

# 5c. Consecutive + Anti-Chop + Body Dominance (top 3 expected)
run_test "05c_consec_antichop_body" \
    "--all ${DAYS} --scalp --micro-regime \
    --override scalp_consecutive_candles_enabled=true \
    --override scalp_anti_chop_enabled=true \
    --override scalp_body_dominance_enabled=true \
    --override scalp_micro_range_enabled=false \
    --override scalp_momentum_candle_enabled=false \
    --override scalp_rsi_filter_enabled=false \
    --override scalp_two_pole_filter_enabled=false \
    --override scalp_macd_filter_enabled=false \
    --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 6: PARAMETER VARIATIONS FOR CONSECUTIVE CANDLES
# =============================================================================

# 6a. Require 3 consecutive candles instead of 2
run_test "06a_consec_min_3" \
    "--all ${DAYS} --scalp --micro-consec-only \
    --override scalp_consecutive_candles_min=3 \
    --parallel --workers ${WORKERS}"

# 6b. Require only 1 consecutive candle (more permissive)
run_test "06b_consec_min_1" \
    "--all ${DAYS} --scalp --micro-consec-only \
    --override scalp_consecutive_candles_min=1 \
    --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 7: PARAMETER VARIATIONS FOR ANTI-CHOP
# =============================================================================

# 7a. Stricter anti-chop (max 1 alternation)
run_test "07a_antichop_max_1" \
    "--all ${DAYS} --scalp --micro-antichop-only \
    --override scalp_anti_chop_max_alternations=1 \
    --parallel --workers ${WORKERS}"

# 7b. More permissive anti-chop (max 3 alternations)
run_test "07b_antichop_max_3" \
    "--all ${DAYS} --scalp --micro-antichop-only \
    --override scalp_anti_chop_max_alternations=3 \
    --parallel --workers ${WORKERS}"

# 7c. Longer lookback (6 candles instead of 4)
run_test "07c_antichop_lookback_6" \
    "--all ${DAYS} --scalp --micro-antichop-only \
    --override scalp_anti_chop_lookback=6 \
    --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 8: PARAMETER VARIATIONS FOR BODY DOMINANCE
# =============================================================================

# 8a. Higher ratio requirement (1.5x body to wick)
run_test "08a_body_ratio_1.5" \
    "--all ${DAYS} --scalp --micro-body-only \
    --override scalp_body_dominance_ratio=1.5 \
    --parallel --workers ${WORKERS}"

# 8b. Lower ratio requirement (0.8x - more permissive)
run_test "08b_body_ratio_0.8" \
    "--all ${DAYS} --scalp --micro-body-only \
    --override scalp_body_dominance_ratio=0.8 \
    --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 9: PARAMETER VARIATIONS FOR MICRO-RANGE
# =============================================================================

# 9a. Tighter range filter (min 5 pips instead of 3)
run_test "09a_micro_range_5_pips" \
    "--all ${DAYS} --scalp --micro-range-only \
    --override scalp_micro_range_min_pips=5.0 \
    --parallel --workers ${WORKERS}"

# 9b. More permissive (min 2 pips)
run_test "09b_micro_range_2_pips" \
    "--all ${DAYS} --scalp --micro-range-only \
    --override scalp_micro_range_min_pips=2.0 \
    --parallel --workers ${WORKERS}"

# =============================================================================
# TEST 10: HYBRID - MICRO-REGIME + MOMENTUM FILTERS
# =============================================================================

# 10a. Consecutive Candles + RSI (most promising momentum filter)
run_test "10a_consec_plus_rsi" \
    "--all ${DAYS} --scalp --micro-regime \
    --override scalp_consecutive_candles_enabled=true \
    --override scalp_anti_chop_enabled=false \
    --override scalp_body_dominance_enabled=false \
    --override scalp_micro_range_enabled=false \
    --override scalp_momentum_candle_enabled=false \
    --override scalp_rsi_filter_enabled=true \
    --override scalp_two_pole_filter_enabled=false \
    --override scalp_macd_filter_enabled=false \
    --parallel --workers ${WORKERS}"

# 10b. Best micro-regime + All momentum filters
run_test "10b_consec_antichop_plus_all_momentum" \
    "--all ${DAYS} --scalp --micro-regime \
    --override scalp_consecutive_candles_enabled=true \
    --override scalp_anti_chop_enabled=true \
    --override scalp_body_dominance_enabled=false \
    --override scalp_micro_range_enabled=false \
    --override scalp_momentum_candle_enabled=false \
    --override scalp_rsi_filter_enabled=true \
    --override scalp_two_pole_filter_enabled=true \
    --override scalp_macd_filter_enabled=true \
    --parallel --workers ${WORKERS}"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

log ""
log "${GREEN}============================================================${NC}"
log "${GREEN}ALL TESTS COMPLETED${NC}"
log "${GREEN}============================================================${NC}"
log "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
log ""
log "Results saved to: ${RESULTS_DIR}"
log ""
log "Individual test outputs:"
ls -la "${RESULTS_DIR}/"*.txt 2>/dev/null | tee -a "${LOG_FILE}"
log ""

# Generate comparison summary
log "${YELLOW}============================================================${NC}"
log "${YELLOW}COMPARISON SUMMARY${NC}"
log "${YELLOW}============================================================${NC}"

# Extract key metrics from each test for easy comparison
log ""
log "Test Name | Signals | Win Rate | Total Pips | Profit Factor"
log "----------|---------|----------|------------|---------------"

for output_file in "${RESULTS_DIR}"/*.txt; do
    test_name=$(basename "${output_file}" .txt)

    # Try to extract metrics from the summary
    signals=$(grep -oP "Total Signals.*?(\d+)" "${output_file}" 2>/dev/null | grep -oP "\d+" | tail -1 || echo "N/A")
    win_rate=$(grep -oP "Win Rate.*?(\d+\.?\d*%)" "${output_file}" 2>/dev/null | grep -oP "\d+\.?\d*%" | tail -1 || echo "N/A")
    total_pips=$(grep -oP "Total Pips.*?([+-]?\d+\.?\d*)" "${output_file}" 2>/dev/null | grep -oP "[+-]?\d+\.?\d*" | tail -1 || echo "N/A")
    pf=$(grep -oP "Profit Factor.*?(\d+\.?\d*)" "${output_file}" 2>/dev/null | grep -oP "\d+\.?\d*" | tail -1 || echo "N/A")

    log "${test_name} | ${signals} | ${win_rate} | ${total_pips} | ${pf}"
done | tee -a "${LOG_FILE}"

log ""
log "${GREEN}Test suite complete! Review ${LOG_FILE} for full details.${NC}"
log ""
log "To download results to local machine:"
log "  scp -i ~/.ssh/azure_backtest_rsa -r azureuser@\$(./scripts/azure_backtest.sh vm-ip):${RESULTS_DIR} ./micro_regime_results/"
