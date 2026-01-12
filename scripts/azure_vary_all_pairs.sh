#!/bin/bash
# =============================================================================
# Azure Parameter Variation Test - All Pairs
# Runs parameter variation for each pair sequentially with 14 workers
# =============================================================================

set -euo pipefail

# Configuration
DAYS="${1:-14}"
WORKERS="${2:-14}"
TOP_N="${3:-15}"

# All supported pairs
PAIRS=(
    "EURUSD"
    "GBPUSD"
    "USDJPY"
    "AUDUSD"
    "USDCAD"
    "NZDUSD"
    "EURJPY"
    "AUDJPY"
    "GBPJPY"
    "USDCHF"
)

# Parameter variation JSON
VARY_JSON='{"swing_proximity_min_distance_pips": [8, 10, 12, 14, 16, 18], "fib_pullback_min": [0.2, 0.25, 0.3, 0.35], "fib_pullback_max": [0.6, 0.65, 0.7, 0.75]}'

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Parameter Variation Test - All Pairs ===${NC}"
echo "Days: ${DAYS}"
echo "Workers: ${WORKERS}"
echo "Top N: ${TOP_N}"
echo "Combinations per pair: 96 (6 x 4 x 4)"
echo ""
echo "Parameters being varied:"
echo "  - swing_proximity_min_distance_pips: [8, 10, 12, 14, 16, 18]"
echo "  - fib_pullback_min: [0.2, 0.25, 0.3, 0.35]"
echo "  - fib_pullback_max: [0.6, 0.65, 0.7, 0.75]"
echo ""

# Results file
RESULTS_FILE="/tmp/variation_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Header for results file
echo "=== Parameter Variation Results ===" > "${RESULTS_FILE}"
echo "Date: $(date)" >> "${RESULTS_FILE}"
echo "Days: ${DAYS}" >> "${RESULTS_FILE}"
echo "Parameters: ${VARY_JSON}" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

for pair in "${PAIRS[@]}"; do
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${BLUE}Testing ${pair}...${NC}"
    echo -e "${YELLOW}========================================${NC}"

    echo "" >> "${RESULTS_FILE}"
    echo "========================================" >> "${RESULTS_FILE}"
    echo "PAIR: ${pair}" >> "${RESULTS_FILE}"
    echo "========================================" >> "${RESULTS_FILE}"

    # Run the variation test
    docker exec task-worker python -u /app/forex_scanner/bt.py \
        "${pair}" "${DAYS}" SMC_SIMPLE \
        --timeframe 15m \
        --vary-json "${VARY_JSON}" \
        --vary-workers "${WORKERS}" \
        --rank-by composite_score \
        --top-n "${TOP_N}" 2>&1 | tee -a "${RESULTS_FILE}"

    echo ""
    echo -e "${GREEN}Completed ${pair}${NC}"
    echo ""
done

echo ""
echo -e "${GREEN}=== All Pairs Complete ===${NC}"
echo "Full results saved to: ${RESULTS_FILE}"
echo ""
echo "To view results: cat ${RESULTS_FILE}"
