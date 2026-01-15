#!/bin/bash
# =============================================================================
# Azure Scalp Mode Parallel Parameter Variation
# Uses --vary-json for efficient parallel testing of scalp tier settings
# =============================================================================

set -euo pipefail

# Configuration
PAIR="${1:-EURUSD}"
DAYS="${2:-7}"
WORKERS="${3:-14}"
TOP_N="${4:-20}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parameter variation JSON for scalp mode
# Tests EMA period, swing lookback, and order offset combinations
VARY_JSON='{
    "scalp_ema_period": [10, 15, 20, 30],
    "scalp_swing_lookback_bars": [6, 8, 10, 12, 15],
    "scalp_limit_offset_pips": [1, 2, 3]
}'

# Calculate combinations
COMBINATIONS=$((4 * 5 * 3))

echo -e "${BLUE}=== Scalp Mode Parallel Parameter Variation ===${NC}"
echo ""
echo "Pair: ${PAIR}"
echo "Days: ${DAYS}"
echo "Workers: ${WORKERS}"
echo "Top N: ${TOP_N}"
echo ""
echo "Parameter Variations:"
echo "  - scalp_ema_period: [10, 15, 20, 30]"
echo "  - scalp_swing_lookback_bars: [6, 8, 10, 12, 15]"
echo "  - scalp_limit_offset_pips: [1, 2, 3]"
echo ""
echo "Total combinations: ${COMBINATIONS}"
echo ""

# Results file
RESULTS_FILE="/tmp/scalp_parallel_${PAIR}_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Header
{
    echo "=== Scalp Mode Parallel Variation Results ==="
    echo "Date: $(date)"
    echo "Pair: ${PAIR}"
    echo "Days: ${DAYS}"
    echo "Workers: ${WORKERS}"
    echo "Combinations: ${COMBINATIONS}"
    echo ""
} > "${RESULTS_FILE}"

echo -e "${YELLOW}Starting parallel variation test...${NC}"
echo ""

# Run the variation test
docker exec task-worker python -u /app/forex_scanner/bt.py \
    "${PAIR}" "${DAYS}" SMC_SIMPLE \
    --scalp \
    --vary-json "${VARY_JSON}" \
    --vary-workers "${WORKERS}" \
    --rank-by composite_score \
    --top-n "${TOP_N}" 2>&1 | tee -a "${RESULTS_FILE}"

echo ""
echo -e "${GREEN}=== Test Complete ===${NC}"
echo "Results saved to: ${RESULTS_FILE}"
