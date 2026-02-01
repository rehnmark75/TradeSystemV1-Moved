#!/bin/bash
# =============================================================================
# Regime-Based Parameter Optimization Script for Azure VM
# =============================================================================
#
# This script runs comprehensive parameter optimization tests during regime-
# dominated periods to find optimal trading parameters for each market condition.
#
# Usage:
#   ./regime_optimization_azure.sh [EPIC] [REGIME] [DAYS]
#
# Examples:
#   ./regime_optimization_azure.sh                           # Default: EURUSD, all regimes, 180 days
#   ./regime_optimization_azure.sh EURUSD trending 180       # Trending periods for EURUSD
#   ./regime_optimization_azure.sh GBPUSD high_volatility 90 # High vol periods for GBPUSD
#
# Prerequisites:
#   - Docker containers running (task-worker, postgres)
#   - Market intelligence history table populated
#   - Candle data available for the test period
#
# =============================================================================

set -e  # Exit on error

# Configuration
EPIC=${1:-EURUSD}
TARGET_REGIME=${2:-all}
DAYS=${3:-180}
RESULTS_DIR="/tmp/regime_optimization_${EPIC}_$(date +%Y%m%d_%H%M%S)"
MIN_REGIME_PCT=50  # Minimum % for a period to be considered dominated
MIN_PERIOD_DAYS=5  # Minimum days for a valid period

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "REGIME PARAMETER OPTIMIZATION"
echo "==========================================${NC}"
echo ""
echo "Epic:           ${EPIC}"
echo "Target Regime:  ${TARGET_REGIME}"
echo "Analysis Days:  ${DAYS}"
echo "Results Dir:    ${RESULTS_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# =============================================================================
# Step 1: Analyze Regime Distribution
# =============================================================================
echo -e "${YELLOW}Step 1: Analyzing regime distribution...${NC}"

docker exec postgres psql -U postgres -d forex -c "
WITH regime_by_day AS (
    SELECT
        scan_timestamp::date as day,
        dominant_regime,
        COUNT(*) as count
    FROM market_intelligence_history
    WHERE scan_timestamp >= NOW() - INTERVAL '${DAYS} days'
    GROUP BY 1, 2
),
daily_totals AS (
    SELECT day, SUM(count) as total FROM regime_by_day GROUP BY day
),
daily_pct AS (
    SELECT
        r.day,
        r.dominant_regime,
        r.count * 100.0 / t.total as pct
    FROM regime_by_day r
    JOIN daily_totals t ON r.day = t.day
)
SELECT
    dominant_regime as regime,
    COUNT(*) as days_present,
    ROUND(AVG(pct), 1) as avg_pct_when_present,
    COUNT(*) FILTER (WHERE pct >= ${MIN_REGIME_PCT}) as days_dominant
FROM daily_pct
GROUP BY dominant_regime
ORDER BY days_dominant DESC;
" | tee "${RESULTS_DIR}/regime_distribution.txt"

echo ""

# =============================================================================
# Step 2: Find Regime-Dominated Periods
# =============================================================================
echo -e "${YELLOW}Step 2: Finding regime-dominated periods...${NC}"

# Run the period analyzer
docker exec task-worker python /app/forex_scanner/scripts/regime_period_analyzer.py \
    --epic ${EPIC} \
    --all-regimes \
    --days ${DAYS} \
    --min-pct ${MIN_REGIME_PCT} \
    --min-days ${MIN_PERIOD_DAYS} \
    --output-json > "${RESULTS_DIR}/regime_periods.json"

echo "Found periods saved to: ${RESULTS_DIR}/regime_periods.json"
cat "${RESULTS_DIR}/regime_periods.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
periods = data.get('dominated_periods', [])
print(f'Found {len(periods)} regime-dominated periods:')
for p in periods:
    print(f\"  - {p['regime']}: {p['start_date']} to {p['end_date']} ({p['days']} days, {p['regime_pct']:.0f}%)\")
"

echo ""

# =============================================================================
# Step 3: Define Parameter Grid
# =============================================================================
echo -e "${YELLOW}Step 3: Defining parameter grid...${NC}"

# Parameter grids for each regime
declare -A PARAM_GRIDS

# Trending: Wider TP, moderate SL, lower confidence (trends are clear)
PARAM_GRIDS[trending]='{
  "fixed_stop_loss_pips": [8, 10, 12],
  "fixed_take_profit_pips": [15, 18, 20],
  "min_confidence": [0.40, 0.45, 0.50]
}'

# Ranging: Tight SL/TP, higher confidence (need clear signals)
PARAM_GRIDS[ranging]='{
  "fixed_stop_loss_pips": [10, 12, 15],
  "fixed_take_profit_pips": [10, 12, 15],
  "min_confidence": [0.50, 0.55, 0.60]
}'

# High Volatility: Wide SL, tight TP, high confidence
PARAM_GRIDS[high_volatility]='{
  "fixed_stop_loss_pips": [12, 15, 18],
  "fixed_take_profit_pips": [8, 10, 12],
  "min_confidence": [0.55, 0.60, 0.65]
}'

# Low Volatility: Moderate SL/TP
PARAM_GRIDS[low_volatility]='{
  "fixed_stop_loss_pips": [8, 10, 12],
  "fixed_take_profit_pips": [12, 15, 18],
  "min_confidence": [0.45, 0.50, 0.55]
}'

# Breakout: Wider TP for momentum
PARAM_GRIDS[breakout]='{
  "fixed_stop_loss_pips": [10, 12, 15],
  "fixed_take_profit_pips": [15, 20, 25],
  "min_confidence": [0.50, 0.55, 0.60]
}'

# Reversal: Moderate settings
PARAM_GRIDS[reversal]='{
  "fixed_stop_loss_pips": [10, 12, 15],
  "fixed_take_profit_pips": [12, 15, 18],
  "min_confidence": [0.55, 0.60, 0.65]
}'

echo "Parameter grids defined for: trending, ranging, high_volatility, low_volatility, breakout, reversal"

# =============================================================================
# Step 4: Run Optimization Tests
# =============================================================================
echo -e "${YELLOW}Step 4: Running optimization tests...${NC}"
echo ""

# Function to run optimization for a regime
run_regime_optimization() {
    local regime=$1
    local grid="${PARAM_GRIDS[$regime]}"

    if [ -z "$grid" ]; then
        echo -e "${RED}No parameter grid defined for regime: $regime${NC}"
        return 1
    fi

    echo -e "${GREEN}Optimizing for ${regime}...${NC}"

    # Save grid to file
    echo "$grid" > "${RESULTS_DIR}/grid_${regime}.json"

    # Run optimization
    docker exec task-worker python /app/forex_scanner/scripts/regime_optimization_runner.py \
        --epic ${EPIC} \
        --regime ${regime} \
        --days ${DAYS} \
        --min-pct ${MIN_REGIME_PCT} \
        --grid "/app/forex_scanner/scripts/../../../tmp/regime_optimization_${EPIC}_*/grid_${regime}.json" \
        --max-combinations 50 \
        --output-json 2>&1 | tee "${RESULTS_DIR}/optimization_${regime}.json"

    echo ""
}

# Run for target regime or all regimes
if [ "$TARGET_REGIME" == "all" ]; then
    for regime in trending ranging high_volatility low_volatility breakout reversal; do
        # Check if we have periods for this regime
        PERIOD_COUNT=$(cat "${RESULTS_DIR}/regime_periods.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
periods = [p for p in data.get('dominated_periods', []) if p['regime'] == '${regime}']
print(len(periods))
")
        if [ "$PERIOD_COUNT" -gt 0 ]; then
            run_regime_optimization "$regime"
        else
            echo -e "${YELLOW}Skipping ${regime}: No dominated periods found${NC}"
        fi
    done
else
    run_regime_optimization "$TARGET_REGIME"
fi

# =============================================================================
# Step 5: Aggregate Results
# =============================================================================
echo -e "${YELLOW}Step 5: Aggregating results...${NC}"

# Create summary file
SUMMARY_FILE="${RESULTS_DIR}/OPTIMIZATION_SUMMARY.txt"

cat > "${SUMMARY_FILE}" << EOF
================================================================================
REGIME PARAMETER OPTIMIZATION RESULTS
================================================================================
Epic:           ${EPIC}
Analysis Days:  ${DAYS}
Generated:      $(date)

--------------------------------------------------------------------------------
OPTIMAL PARAMETERS BY REGIME
--------------------------------------------------------------------------------

EOF

# Parse each optimization result and extract best parameters
for regime in trending ranging high_volatility low_volatility breakout reversal; do
    RESULT_FILE="${RESULTS_DIR}/optimization_${regime}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "Processing $regime results..."

        # Extract best result using Python
        python3 << PYEOF >> "${SUMMARY_FILE}"
import json

try:
    with open('${RESULT_FILE}') as f:
        content = f.read()
        # Find JSON object in output
        start = content.find('{')
        if start >= 0:
            data = json.loads(content[start:])
            for run in data.get('results', []):
                best = run.get('best', {})
                if best:
                    params = best.get('params', {})
                    print(f"\n${regime^^}:")
                    print(f"  Period: {run.get('period', 'N/A')}")
                    print(f"  Parameters:")
                    for k, v in params.items():
                        print(f"    - {k}: {v}")
                    print(f"  Performance:")
                    print(f"    - Signals: {best.get('signals', 0)}")
                    print(f"    - Win Rate: {best.get('win_rate', 0):.1f}%")
                    print(f"    - Profit Factor: {best.get('profit_factor', 0):.2f}")
                    print(f"    - Expectancy: {best.get('expectancy', 0):+.2f} pips/trade")
except Exception as e:
    print(f"\n${regime^^}: Failed to parse - {e}")
PYEOF
    fi
done

cat >> "${SUMMARY_FILE}" << EOF

--------------------------------------------------------------------------------
RECOMMENDED CONFIGURATION
--------------------------------------------------------------------------------

Based on optimization results, consider adding these regime-specific parameters
to the smc_simple_pair_overrides.parameter_overrides JSONB column:

{
  "regime_params": {
    "trending": {"sl": X, "tp": Y, "conf": Z},
    "ranging": {"sl": X, "tp": Y, "conf": Z},
    "high_volatility": {"sl": X, "tp": Y, "conf": Z},
    "low_volatility": {"sl": X, "tp": Y, "conf": Z}
  }
}

NOTE: Dynamic regime switching requires implementing the parameter loader in
smc_simple_strategy.py to read current regime and apply matching parameters.

================================================================================
EOF

echo ""
echo -e "${GREEN}=========================================="
echo "OPTIMIZATION COMPLETE"
echo "==========================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Files:"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Summary:"
cat "${SUMMARY_FILE}"
