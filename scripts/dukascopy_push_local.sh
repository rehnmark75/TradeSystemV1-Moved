#!/usr/bin/env bash
# =============================================================================
# Dukascopy -> ig_candles_backtest (LOCAL postgres, no Azure)
# =============================================================================
#
# Loads Dukascopy 1m CSVs (produced by scripts/dukascopy_download.py) directly
# into the local postgres `ig_candles_backtest` table, then self-resamples
# to 5m / 15m / 1h / 4h within the same table.
#
# Intentionally does NOT touch `ig_candles` — that table is the live scanner's
# source of truth and gets backed up. `ig_candles_backtest` is derived data
# excluded from backups (see scripts/enhanced_backup.sh), and this script
# IS the authoritative re-population mechanism for disaster recovery.
#
# Usage:
#   ./scripts/dukascopy_push_local.sh /tmp/dukas/
#   ./scripts/dukascopy_push_local.sh /tmp/dukas/ --dry-run   # just show counts
#   ./scripts/dukascopy_push_local.sh /tmp/dukas/ --skip-resample   # 1m only
#
# =============================================================================

set -euo pipefail

CSV_DIR="${1:-}"
DRY_RUN=0
SKIP_RESAMPLE=0
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-postgres}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --skip-resample) SKIP_RESAMPLE=1 ;;
    esac
done

if [[ -z "${CSV_DIR}" || ! -d "${CSV_DIR}" ]]; then
    echo -e "${RED}Usage: $0 <csv_dir> [--dry-run] [--skip-resample]${NC}"
    echo ""
    echo "Example:"
    echo "  $0 /tmp/dukas/"
    exit 1
fi

# Count CSV files
CSV_FILES=( "${CSV_DIR%/}"/*.csv )
if [[ ${#CSV_FILES[@]} -eq 0 || ! -f "${CSV_FILES[0]}" ]]; then
    echo -e "${RED}ERROR: no CSV files found in ${CSV_DIR}${NC}"
    exit 1
fi

echo -e "${BLUE}=== Dukascopy -> ig_candles_backtest (local) ===${NC}"
echo "CSV dir:   ${CSV_DIR}"
echo "Container: ${POSTGRES_CONTAINER}"
echo "CSV count: ${#CSV_FILES[@]}"
[[ ${DRY_RUN}        -eq 1 ]] && echo -e "${YELLOW}DRY RUN${NC}"
[[ ${SKIP_RESAMPLE}  -eq 1 ]] && echo -e "${YELLOW}SKIP RESAMPLE${NC}"
echo ""

# =============================================================================
# Sanity checks
# =============================================================================

if ! docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
    echo -e "${RED}ERROR: container '${POSTGRES_CONTAINER}' is not running${NC}"
    exit 1
fi

# Make sure target table exists (fresh-environment safety net).
# In normal operation it always does, but we don't want to silently fail
# a restore-from-scratch procedure.
echo -e "${BLUE}[1/4] Ensuring ig_candles_backtest exists...${NC}"
docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -q -c "
CREATE TABLE IF NOT EXISTS ig_candles_backtest (
    start_time     TIMESTAMP NOT NULL,
    epic           VARCHAR NOT NULL,
    timeframe      INTEGER NOT NULL,
    open           DOUBLE PRECISION NOT NULL,
    high           DOUBLE PRECISION NOT NULL,
    low            DOUBLE PRECISION NOT NULL,
    close          DOUBLE PRECISION NOT NULL,
    volume         INTEGER NOT NULL,
    ltv            INTEGER,
    resampled_from INTEGER DEFAULT 1,
    created_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (start_time, epic, timeframe)
);
CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic ON ig_candles_backtest(epic);
CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic_tf_time ON ig_candles_backtest(epic, timeframe, start_time DESC);
"

# =============================================================================
# Load each CSV into a per-file temp table, then INSERT ... ON CONFLICT DO NOTHING
# =============================================================================

echo -e "${BLUE}[2/4] Loading 1m CSVs into ig_candles_backtest...${NC}"

TOTAL_LOADED=0
for csv in "${CSV_FILES[@]}"; do
    csv_base="$(basename "${csv}")"
    lines=$(( $(wc -l < "${csv}") - 1 ))
    echo -e "  ${csv_base}  (${lines} rows)"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        continue
    fi

    # Copy CSV into container (postgres container doesn't see host paths)
    docker cp "${csv}" "${POSTGRES_CONTAINER}:/tmp/_dukas_staging.csv"

    # Load into staging table, then INSERT ... ON CONFLICT DO NOTHING.
    # NOTE: `docker exec` needs `-i` for heredoc stdin to reach psql —
    # without it psql sees empty input and silently no-ops.
    inserted=$(docker exec -i "${POSTGRES_CONTAINER}" psql -U postgres -d forex -v ON_ERROR_STOP=1 -t -A <<'SQL'
DROP TABLE IF EXISTS _dukas_stage;
CREATE TABLE _dukas_stage (
    start_time     TIMESTAMP NOT NULL,
    epic           VARCHAR NOT NULL,
    timeframe      INTEGER NOT NULL,
    open           DOUBLE PRECISION NOT NULL,
    high           DOUBLE PRECISION NOT NULL,
    low            DOUBLE PRECISION NOT NULL,
    close          DOUBLE PRECISION NOT NULL,
    volume         INTEGER NOT NULL,
    ltv            INTEGER,
    resampled_from INTEGER
);

\COPY _dukas_stage FROM '/tmp/_dukas_staging.csv' WITH CSV HEADER

WITH ins AS (
    INSERT INTO ig_candles_backtest
        (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
    SELECT start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from
      FROM _dukas_stage
    ON CONFLICT (start_time, epic, timeframe) DO NOTHING
    RETURNING 1
)
SELECT COUNT(*) FROM ins;

DROP TABLE _dukas_stage;
SQL
)
    docker exec "${POSTGRES_CONTAINER}" rm -f /tmp/_dukas_staging.csv
    # inserted may contain multiple lines due to DROP TABLE notices; grab the numeric line
    inserted_n=$(echo "${inserted}" | grep -E '^[0-9]+$' | head -1)
    inserted_n=${inserted_n:-0}
    echo "    inserted ${inserted_n} new rows"
    TOTAL_LOADED=$((TOTAL_LOADED + inserted_n))
done

echo -e "${GREEN}  Loaded ~${TOTAL_LOADED} rows from ${#CSV_FILES[@]} files${NC}"

# =============================================================================
# Self-resample 1m → 5m / 15m / 1h / 4h within ig_candles_backtest
# =============================================================================

if [[ ${SKIP_RESAMPLE} -eq 1 || ${DRY_RUN} -eq 1 ]]; then
    echo -e "${YELLOW}[3/4] Resample skipped${NC}"
else
    echo -e "${BLUE}[3/4] Resampling 1m -> 5m / 15m / 1h / 4h...${NC}"
    echo "  (only resampling epics present in the 1m data, to avoid touching pre-existing rows from other sources)"

    # Get list of epics we just loaded (have 1m data)
    EPICS=$(docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -At -c "
SELECT DISTINCT epic FROM ig_candles_backtest WHERE timeframe = 1
")

    for epic in ${EPICS}; do
        echo "  ${epic}:"
        # 5m
        docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -q -c "
INSERT INTO ig_candles_backtest (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
SELECT
    date_trunc('hour', start_time) + INTERVAL '5 min' * FLOOR(EXTRACT(minute FROM start_time) / 5) AS start_time,
    epic, 5,
    (array_agg(open  ORDER BY start_time))[1],
    MAX(high), MIN(low),
    (array_agg(close ORDER BY start_time DESC))[1],
    SUM(volume), SUM(COALESCE(ltv, 0)), 1
FROM ig_candles_backtest
WHERE timeframe = 1 AND epic = '${epic}'
GROUP BY epic, date_trunc('hour', start_time) + INTERVAL '5 min' * FLOOR(EXTRACT(minute FROM start_time) / 5)
ON CONFLICT (start_time, epic, timeframe) DO NOTHING;
"
        # 15m
        docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -q -c "
INSERT INTO ig_candles_backtest (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
SELECT
    date_trunc('hour', start_time) + INTERVAL '15 min' * FLOOR(EXTRACT(minute FROM start_time) / 15) AS start_time,
    epic, 15,
    (array_agg(open  ORDER BY start_time))[1],
    MAX(high), MIN(low),
    (array_agg(close ORDER BY start_time DESC))[1],
    SUM(volume), SUM(COALESCE(ltv, 0)), 1
FROM ig_candles_backtest
WHERE timeframe = 1 AND epic = '${epic}'
GROUP BY epic, date_trunc('hour', start_time) + INTERVAL '15 min' * FLOOR(EXTRACT(minute FROM start_time) / 15)
ON CONFLICT (start_time, epic, timeframe) DO NOTHING;
"
        # 1h
        docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -q -c "
INSERT INTO ig_candles_backtest (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
SELECT
    date_trunc('hour', start_time) AS start_time,
    epic, 60,
    (array_agg(open  ORDER BY start_time))[1],
    MAX(high), MIN(low),
    (array_agg(close ORDER BY start_time DESC))[1],
    SUM(volume), SUM(COALESCE(ltv, 0)), 1
FROM ig_candles_backtest
WHERE timeframe = 1 AND epic = '${epic}'
GROUP BY epic, date_trunc('hour', start_time)
ON CONFLICT (start_time, epic, timeframe) DO NOTHING;
"
        # 4h
        docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -q -c "
INSERT INTO ig_candles_backtest (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
SELECT
    date_trunc('day', start_time) + INTERVAL '4 hour' * FLOOR(EXTRACT(hour FROM start_time) / 4) AS start_time,
    epic, 240,
    (array_agg(open  ORDER BY start_time))[1],
    MAX(high), MIN(low),
    (array_agg(close ORDER BY start_time DESC))[1],
    SUM(volume), SUM(COALESCE(ltv, 0)), 1
FROM ig_candles_backtest
WHERE timeframe = 1 AND epic = '${epic}'
GROUP BY epic, date_trunc('day', start_time) + INTERVAL '4 hour' * FLOOR(EXTRACT(hour FROM start_time) / 4)
ON CONFLICT (start_time, epic, timeframe) DO NOTHING;
"
        echo "    ✓ resampled to 5m / 15m / 60m / 240m"
    done
fi

# =============================================================================
# Summary
# =============================================================================

echo -e "${BLUE}[4/4] Summary${NC}"
docker exec "${POSTGRES_CONTAINER}" psql -U postgres -d forex -c "
SELECT epic, timeframe, COUNT(*) AS rows,
       MIN(start_time)::date AS earliest,
       MAX(start_time)::date AS latest
FROM ig_candles_backtest
WHERE epic IN (
    'CS.D.EURUSD.CEEM.IP','CS.D.GBPUSD.MINI.IP','CS.D.USDJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP','CS.D.USDCHF.MINI.IP','CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP','CS.D.EURJPY.MINI.IP','CS.D.AUDJPY.MINI.IP',
    'CS.D.GBPJPY.MINI.IP','CS.D.EURGBP.MINI.IP','CS.D.CFEGOLD.DUKAS.IP'
)
GROUP BY epic, timeframe
ORDER BY epic, timeframe;
"

echo ""
echo -e "${GREEN}=== Done ===${NC}"
echo ""
echo "Verify with:"
echo "  docker exec postgres psql -U postgres -d forex -c \"SELECT pg_size_pretty(pg_total_relation_size('ig_candles_backtest'));\""
