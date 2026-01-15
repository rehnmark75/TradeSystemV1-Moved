#!/bin/bash
# =============================================================================
# Azure Scalp Mode Parameter Variation Test
# Tests different scalp tier settings across pairs
# =============================================================================

set -euo pipefail

# Configuration
DAYS="${1:-7}"
WORKERS="${2:-14}"

# Pairs to test
PAIRS=(
    "EURUSD"
    "GBPUSD"
    "USDJPY"
    "AUDUSD"
    "NZDUSD"
    "EURJPY"
    "GBPJPY"
    "AUDJPY"
)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parameter variations
EMA_PERIODS=(10 15 20 30)
SWING_LOOKBACKS=(6 8 10 12 15)
OFFSETS=(1 2 3)

# Results file
RESULTS_FILE="/tmp/scalp_vary_results_$(date +%Y%m%d_%H%M%S).txt"

echo -e "${BLUE}=== Scalp Mode Parameter Variation Test ===${NC}"
echo "Days: ${DAYS}"
echo "Workers: ${WORKERS}"
echo "Pairs: ${PAIRS[*]}"
echo ""
echo "Parameters being varied:"
echo "  - EMA Period: ${EMA_PERIODS[*]}"
echo "  - Swing Lookback: ${SWING_LOOKBACKS[*]}"
echo "  - Order Offset: ${OFFSETS[*]}"
echo ""
echo "Total combinations per pair: $((${#EMA_PERIODS[@]} * ${#SWING_LOOKBACKS[@]} * ${#OFFSETS[@]}))"
echo "Results file: ${RESULTS_FILE}"
echo ""

# Header for results file
{
    echo "=== Scalp Mode Parameter Variation Results ==="
    echo "Date: $(date)"
    echo "Days: ${DAYS}"
    echo "EMA Periods: ${EMA_PERIODS[*]}"
    echo "Swing Lookbacks: ${SWING_LOOKBACKS[*]}"
    echo "Offsets: ${OFFSETS[*]}"
    echo ""
    echo "Format: PAIR | EMA | SWING | OFFSET | SIGNALS | WIN_RATE | PROFIT_FACTOR | EXPECTANCY"
    echo "========================================================================"
} > "${RESULTS_FILE}"

for pair in "${PAIRS[@]}"; do
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${BLUE}Testing ${pair}...${NC}"
    echo -e "${YELLOW}========================================${NC}"

    echo "" >> "${RESULTS_FILE}"
    echo "========================================" >> "${RESULTS_FILE}"
    echo "PAIR: ${pair}" >> "${RESULTS_FILE}"
    echo "========================================" >> "${RESULTS_FILE}"

    for ema in "${EMA_PERIODS[@]}"; do
        for swing in "${SWING_LOOKBACKS[@]}"; do
            for offset in "${OFFSETS[@]}"; do
                echo "  Testing EMA=${ema}, Swing=${swing}, Offset=${offset}..."

                # Run the backtest and capture output
                OUTPUT=$(docker exec task-worker python -u /app/forex_scanner/bt.py \
                    "${pair}" "${DAYS}" \
                    --scalp \
                    --scalp-ema "${ema}" \
                    --scalp-swing-lookback "${swing}" \
                    --scalp-offset "${offset}" 2>&1)

                # Extract key metrics from output
                SIGNALS=$(echo "${OUTPUT}" | grep -oP "Total Signals: \K\d+" | tail -1 || echo "0")
                WIN_RATE=$(echo "${OUTPUT}" | grep -oP "Win Rate: \K[\d.]+%" | tail -1 || echo "0%")
                PROFIT_FACTOR=$(echo "${OUTPUT}" | grep -oP "Profit Factor: \K[\d.]+" | tail -1 || echo "0")
                EXPECTANCY=$(echo "${OUTPUT}" | grep -oP "Expectancy: \K[\d.-]+ pips" | tail -1 || echo "0 pips")

                # Log result
                RESULT_LINE="${pair} | EMA=${ema} | SWING=${swing} | OFFSET=${offset} | ${SIGNALS} signals | WR=${WIN_RATE} | PF=${PROFIT_FACTOR} | EXP=${EXPECTANCY}"
                echo "    -> ${WIN_RATE} win rate, ${SIGNALS} signals"
                echo "${RESULT_LINE}" >> "${RESULTS_FILE}"
            done
        done
    done

    echo -e "${GREEN}Completed ${pair}${NC}"
    echo ""
done

echo ""
echo -e "${GREEN}=== All Tests Complete ===${NC}"
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "To view results: cat ${RESULTS_FILE}"
echo ""
echo "To find best combinations:"
echo "  grep -E 'Win Rate|EMA=' ${RESULTS_FILE} | sort -t'=' -k4 -rn"
