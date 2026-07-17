#!/usr/bin/env python3
"""
Candle snapshotter — pushes the tail of the local 1-minute candle store to Azure
Blob so the trading Function can read recent price action WITHOUT touching IG's
historical-price allowance (which is shared with the streaming system).

Runs next to the Postgres candle DB at home. Outbound HTTPS only — Azure never
reaches into the home network. Each cycle it OVERWRITES one small JSON blob per
epic; the blob's `generatedAt` lets the Function detect a dead snapshotter and
fail open (skip the plausibility gate rather than judge stale data).

The DB stores only 1-minute bars (ig_candles timeframe=1); higher timeframes are
resampled in SQL via date_bin, matching how the scanner's DataFetcher resamples
in memory. One blob per (epic, series):

    series   bars   span     role
    5min     72     6 h      entry context — is this bar chasing?
    15min    96     24 h     primary swing structure (majority strategy TF)
    1h       48     2 days   trend / bias (native for the rare 1h strategies)

Blob layout (container `candles`, keyed by epic only — candle data is the same on
demo and live):
    CS.D.CFEGOLD.CEE.IP/5min.json   (also 15min.json, 1h.json)
    {
      "epic": "CS.D.CFEGOLD.CEE.IP",
      "resolution": "5min",
      "generatedAt": "2026-07-09T09:32:00Z",
      "candles": [ {"t": "...Z", "o": .., "h": .., "l": .., "c": ..}, ... ]  # oldest -> newest
    }
The newest candle is usually STILL FORMING (built from the 1m bars seen so far);
consumers should treat candles[-1] as partial unless its bucket has fully elapsed.

Run modes:
    loop (default) — snapshot every SNAPSHOT_INTERVAL_SECONDS forever (systemd / docker).
    once           — set RUN_ONCE=1 for a single pass (driven by cron).
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2 import sql
from azure.storage.blob import ContainerClient, ContentSettings

# ---------------------------------------------------------------------------
# CONFIG — defaults match TradeSystemV1's `forex.ig_candles` table, where all
# timeframes share one table (timeframe=1 → 1-minute bars) and start_time is a
# naive-UTC timestamp. Everything is overridable via environment variables.
# ---------------------------------------------------------------------------

CANDLES_TABLE = os.environ.get("CANDLES_TABLE", "ig_candles")
COL_EPIC = os.environ.get("COL_EPIC", "epic")
COL_TS = os.environ.get("COL_TS", "start_time")  # candle open time (naive UTC)
COL_OPEN = os.environ.get("COL_OPEN", "open")
COL_HIGH = os.environ.get("COL_HIGH", "high")
COL_LOW = os.environ.get("COL_LOW", "low")
COL_CLOSE = os.environ.get("COL_CLOSE", "close")
COL_TIMEFRAME = os.environ.get("COL_TIMEFRAME", "timeframe")
TIMEFRAME_VALUE = int(os.environ.get("TIMEFRAME_VALUE", "1"))  # 1 = 1-minute bars

# The instruments to snapshot. Non-epic junk (e.g. a stray "c") is filtered out.
ACTIVE_EPICS = [
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.USDCHF.MINI.IP",
    "CS.D.CFEGOLD.CEE.IP",   # Spot Gold ($1)
    "CS.D.CFDSILVER.CFM.IP",  # Silver Mini
]

# Series to publish: "<label>:<bucket-minutes>:<bar-count>" comma-separated.
_SERIES_SPEC = os.environ.get("SERIES", "5min:5:72,15min:15:96,1h:60:48")
SERIES = [
    (label, int(minutes), int(bars))
    for label, minutes, bars in (s.strip().split(":") for s in _SERIES_SPEC.split(",") if s.strip())
]
SNAPSHOT_INTERVAL_SECONDS = int(os.environ.get("SNAPSHOT_INTERVAL_SECONDS", "60"))
RUN_ONCE = os.environ.get("RUN_ONCE", "").lower() in ("1", "true", "yes")

# Postgres connection: prefer a single DSN, else assemble from discrete PG* vars.
PG_DSN = os.environ.get("PG_DSN") or os.environ.get("DATABASE_URL")

# Azure Blob: a CONTAINER-scoped SAS URL with write/create perms (NOT the account key).
#   https://<acct>.blob.core.windows.net/candles?<sas-token>
BLOB_CONTAINER_SAS_URL = os.environ.get("BLOB_CONTAINER_SAS_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("snapshotter")
# The Azure SDK logs every HTTP request/response header at INFO — keep only warnings.
logging.getLogger("azure").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------

# Resample the 1-minute base series into N-minute OHLC buckets, newest first (we
# reverse before writing). date_bin (PG 14+) aligns buckets to a fixed midnight
# origin so 5m/15m/1h bars land on :00/:05/... exactly like the scanner's
# DataFetcher resample. The timeframe predicate is essential on ig_candles: all
# resolutions could share one table. The start_time cutoff keeps the aggregate
# on the PK index instead of grouping the epic's whole history. Table/column
# names are injected as safe identifiers; everything else is a bound param.
_QUERY = sql.SQL(
    "SELECT date_bin(%(bucket)s, {ts}, TIMESTAMP '2000-01-01') AS bucket_ts, "
    "       (array_agg({o} ORDER BY {ts} ASC))[1]  AS o, "
    "       max({h})                               AS h, "
    "       min({l})                               AS l, "
    "       (array_agg({c} ORDER BY {ts} DESC))[1] AS c "
    "FROM {table} "
    "WHERE {epic} = %(epic)s AND {tf} = %(tf)s AND {ts} >= %(cutoff)s "
    "GROUP BY bucket_ts ORDER BY bucket_ts DESC LIMIT %(bars)s"
).format(
    ts=sql.Identifier(COL_TS),
    o=sql.Identifier(COL_OPEN),
    h=sql.Identifier(COL_HIGH),
    l=sql.Identifier(COL_LOW),
    c=sql.Identifier(COL_CLOSE),
    table=sql.Identifier(CANDLES_TABLE),
    epic=sql.Identifier(COL_EPIC),
    tf=sql.Identifier(COL_TIMEFRAME),
)


def _iso_z(dt: datetime) -> str:
    """UTC ISO-8601 with a trailing Z. Naive timestamps are assumed to be UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def valid_epics():
    seen = set()
    out = []
    for e in ACTIVE_EPICS:
        e = (e or "").strip()
        if not e.startswith("CS.") or e in seen:
            if e and not e.startswith("CS."):
                log.warning("skipping non-epic entry %r", e)
            continue
        seen.add(e)
        out.append(e)
    return out


def fetch_candles(cur, epic, bucket_minutes, bars):
    """Return this epic's last `bars` candles at `bucket_minutes`, oldest -> newest.

    Weekend/quiet gaps make bars-per-wall-clock uneven, so the cutoff is padded
    generously (weekend ~= 48h + 25% slack); LIMIT trims back to `bars`.
    """
    span_minutes = bars * bucket_minutes
    cutoff_minutes = int(span_minutes * 1.25) + 48 * 60 + bucket_minutes
    cur.execute(
        _QUERY,
        {
            "bucket": timedelta(minutes=bucket_minutes),
            "epic": epic,
            "tf": TIMEFRAME_VALUE,
            "cutoff": datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=cutoff_minutes),
            "bars": bars,
        },
    )
    rows = cur.fetchall()
    rows.reverse()  # DB gave newest-first; blob is oldest-first
    return [
        {
            "t": _iso_z(ts) if isinstance(ts, datetime) else str(ts),
            "o": float(o),
            "h": float(h),
            "l": float(l),
            "c": float(c),
        }
        for (ts, o, h, l, c) in rows
    ]


def snapshot_once(container: ContainerClient):
    """One pass: read each epic from Postgres and overwrite its blob."""
    epics = valid_epics()
    now_z = _iso_z(datetime.now(timezone.utc))
    conn = psycopg2.connect(PG_DSN)  # reconnect per pass — survives DB restarts
    try:
        wrote = 0
        total = len(epics) * len(SERIES)
        with conn.cursor() as cur:
            for epic in epics:
                for label, minutes, bars in SERIES:
                    try:
                        candles = fetch_candles(cur, epic, minutes, bars)
                        if not candles:
                            log.warning("no candles for %s %s — skipping (blob left stale)", epic, label)
                            continue
                        payload = {
                            "epic": epic,
                            "resolution": label,
                            "generatedAt": now_z,
                            "candles": candles,
                        }
                        container.upload_blob(
                            name=f"{epic}/{label}.json",
                            data=json.dumps(payload, separators=(",", ":")),
                            overwrite=True,
                            content_settings=ContentSettings(content_type="application/json"),
                        )
                        wrote += 1
                    except Exception:
                        # One bad epic/series must not sink the rest of the pass.
                        log.exception("failed to snapshot %s %s", epic, label)
                        conn.rollback()  # clear aborted-transaction state for the next query
        log.info("snapshot pass complete: %d/%d blobs written", wrote, total)
    finally:
        conn.close()


def main():
    missing = [n for n, v in (("PG_DSN", PG_DSN), ("BLOB_CONTAINER_SAS_URL", BLOB_CONTAINER_SAS_URL)) if not v]
    if missing:
        log.error("missing required env: %s", ", ".join(missing))
        sys.exit(2)

    container = ContainerClient.from_container_url(BLOB_CONTAINER_SAS_URL)

    if RUN_ONCE:
        snapshot_once(container)
        return

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    log.info(
        "snapshotter starting: %d epics x %s, every %ds",
        len(valid_epics()),
        ", ".join(f"{label}({bars})" for label, _, bars in SERIES),
        SNAPSHOT_INTERVAL_SECONDS,
    )
    while not stop["flag"]:
        started = time.monotonic()
        try:
            snapshot_once(container)
        except Exception:
            log.exception("snapshot pass failed (will retry next cycle)")
        # Sleep the remainder of the interval, waking promptly on a stop signal.
        elapsed = time.monotonic() - started
        remaining = max(0.0, SNAPSHOT_INTERVAL_SECONDS - elapsed)
        while remaining > 0 and not stop["flag"]:
            nap = min(1.0, remaining)
            time.sleep(nap)
            remaining -= nap
    log.info("snapshotter stopped")


if __name__ == "__main__":
    main()
