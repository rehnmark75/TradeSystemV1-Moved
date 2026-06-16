import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const SYSTEM_MONITOR_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";
const ROBOMARKETS_API_URL = process.env.ROBOMARKETS_API_URL || "https://api.stockstrader.com/api/v1";
const ROBOMARKETS_API_KEY = process.env.ROBOMARKETS_API_KEY || "";
const ROBOMARKETS_ACCOUNT_ID = process.env.ROBOMARKETS_ACCOUNT_ID || "";

type SettingType = "bool" | "int" | "float";

interface NormalizedContainer {
  name: string;
  image?: string;
  state?: string;
  status?: string;
  uptimeSeconds?: number;
  restartCount?: number;
  warnings?: unknown;
  errors?: unknown;
}

const SETTING_DEFS: Record<string, {
  label: string;
  description: string;
  type: SettingType;
  defaultValue: string;
  min?: number;
  max?: number;
}> = {
  AUTO_TRADING_ENABLED: {
    label: "Auto Trading",
    description: "Master switch for the open-window stock auto-trader.",
    type: "bool",
    defaultValue: "true",
  },
  AUTO_TRADING_DRY_RUN: {
    label: "Dry Run",
    description: "Record decisions without sending orders to RoboMarkets.",
    type: "bool",
    defaultValue: "false",
  },
  MAX_ORDER_NOTIONAL_USD: {
    label: "Max Order Size",
    description: "Maximum notional value for each order in USD.",
    type: "float",
    defaultValue: "500",
    min: 1,
    max: 5000,
  },
  MAX_ACTIVE_STOCK_ORDERS: {
    label: "Max Active Orders",
    description: "Cap across submitted orders and open broker trades.",
    type: "int",
    defaultValue: "5",
    min: 1,
    max: 20,
  },
  MAX_ORDERS_PER_RUN: {
    label: "Max Orders Per Run",
    description: "Maximum orders the auto-trader can submit in one daily run.",
    type: "int",
    defaultValue: "5",
    min: 1,
    max: 20,
  },
  AUTO_TRADE_MAX_SPREAD_PCT: {
    label: "Max Spread %",
    description: "Reject candidates with a broker spread above this percentage.",
    type: "float",
    defaultValue: "0.4",
    min: 0,
    max: 5,
  },
  AUTO_TRADE_MIN_SCORE: {
    label: "Min Score",
    description: "Minimum Auto-Trader Candidates score required after the open.",
    type: "float",
    defaultValue: "65",
    min: 0,
    max: 100,
  },
  AUTO_TRADE_MIN_RELATIVE_VOLUME: {
    label: "Min Intraday RVOL",
    description: "Optional live intraday relative volume gate. Leave at 0 because RoboMarkets quotes do not provide volume.",
    type: "float",
    defaultValue: "0",
    min: 0,
    max: 20,
  },
  AUTO_TRADE_REQUIRE_INTRADAY_RVOL: {
    label: "Require Intraday RVOL",
    description: "Keep disabled unless the quote feed provides usable live volume.",
    type: "bool",
    defaultValue: "false",
  },
  AUTO_TRADE_MAX_QUOTE_AGE_MINUTES: {
    label: "Max Quote Age",
    description: "Reject candidates whose broker quote is older than this many minutes.",
    type: "int",
    defaultValue: "3",
    min: 0,
    max: 30,
  },
  AUTO_TRADE_PULLBACK_LIMIT_OFFSET_PCT: {
    label: "Pullback Limit Offset %",
    description: "For Pullback bias, place the buy limit this percentage below the current ask.",
    type: "float",
    defaultValue: "0.3",
    min: 0,
    max: 5,
  },
  AUTO_TRADE_VALIDATE_DELAY_MINUTES: {
    label: "Validate After Open",
    description: "Minutes after market open before candidates are refreshed and validated.",
    type: "int",
    defaultValue: "5",
    min: 0,
    max: 60,
  },
  AUTO_TRADE_START_DELAY_MINUTES: {
    label: "Trade After Open",
    description: "Minutes after market open before valid candidates can be ordered.",
    type: "int",
    defaultValue: "15",
    min: 0,
    max: 90,
  },
  AUTO_TRADE_STOP_AFTER_MINUTES: {
    label: "Stop New Orders After",
    description: "Minutes after market open when the automation stops placing new orders.",
    type: "int",
    defaultValue: "45",
    min: 1,
    max: 180,
  },
  AUTO_TRADE_STOP_LOSS_PCT: {
    label: "Stop Loss Distance %",
    description: "Final stop distance recomputed from the actual broker entry price.",
    type: "float",
    defaultValue: "3.0",
    min: 0.1,
    max: 10,
  },
  AUTO_TRADE_TAKE_PROFIT_PCT: {
    label: "Take Profit Distance %",
    description: "Final target distance recomputed from the actual broker entry price.",
    type: "float",
    defaultValue: "5.0",
    min: 0.1,
    max: 25,
  },
  AUTO_TRADE_MAX_STOP_DISTANCE_PCT: {
    label: "Max Stop Distance %",
    description: "Reject the order if the final stop is farther than this from actual entry.",
    type: "float",
    defaultValue: "3.0",
    min: 0.1,
    max: 10,
  },
  AUTO_TRADE_MAX_RISK_PCT: {
    label: "Max Risk %",
    description: "Reject the order if dollars at risk exceed this percentage of actual notional.",
    type: "float",
    defaultValue: "3.0",
    min: 0.1,
    max: 10,
  },
  AUTO_TRADE_MAX_RISK_USD: {
    label: "Max Risk USD",
    description: "Reject the order if the final stop would risk more than this dollar amount.",
    type: "float",
    defaultValue: "15.0",
    min: 0.5,
    max: 500,
  },
};

async function ensureSettingsTable() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS stock_auto_trade_settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      value_type VARCHAR(20) NOT NULL DEFAULT 'string',
      label TEXT,
      description TEXT,
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  await pool.query(`
    ALTER TABLE stock_auto_trade_candidates
    ADD COLUMN IF NOT EXISTS broker_quote_age_minutes INTEGER
  `).catch(() => null);
  await pool.query(`
    ALTER TABLE stock_auto_trade_candidates
    ADD COLUMN IF NOT EXISTS intraday_relative_volume DECIMAL(12,4)
  `).catch(() => null);

  for (const [key, def] of Object.entries(SETTING_DEFS)) {
    await pool.query(
      `INSERT INTO stock_auto_trade_settings (key, value, value_type, label, description)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (key) DO UPDATE SET
         value_type = EXCLUDED.value_type,
         label = EXCLUDED.label,
         description = EXCLUDED.description`,
      [key, def.defaultValue, def.type, def.label, def.description]
    );
  }
}

function parseSetting(key: string, value: unknown): string {
  const def = SETTING_DEFS[key];
  if (!def) {
    throw new Error(`Unsupported setting: ${key}`);
  }
  if (def.type === "bool") {
    if (typeof value !== "boolean") {
      throw new Error(`${key} must be a boolean`);
    }
    return value ? "true" : "false";
  }

  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${key} must be a number`);
  }
  if (def.type === "int" && !Number.isInteger(parsed)) {
    throw new Error(`${key} must be a whole number`);
  }
  if (def.min !== undefined && parsed < def.min) {
    throw new Error(`${key} must be at least ${def.min}`);
  }
  if (def.max !== undefined && parsed > def.max) {
    throw new Error(`${key} must be at most ${def.max}`);
  }
  return String(parsed);
}

function typedValue(value: string, type: SettingType) {
  if (type === "bool") return value.toLowerCase() === "true";
  if (type === "int") return Number.parseInt(value, 10);
  return Number.parseFloat(value);
}

async function getContainers() {
  try {
    const res = await fetch(`${SYSTEM_MONITOR_URL}/api/v1/containers?include_stopped=true`, {
      cache: "no-store",
      signal: AbortSignal.timeout(6000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    const containers = data.containers ?? data;
    if (!Array.isArray(containers)) return [];
    return containers.map((container: Record<string, unknown>): NormalizedContainer => ({
      name: String(container.name ?? ""),
      image: container.image ? String(container.image) : undefined,
      state: container.status ? String(container.status) : undefined,
      status: container.health_status ? String(container.health_status) : undefined,
      uptimeSeconds: typeof container.uptime_seconds === "number" ? container.uptime_seconds : undefined,
      restartCount: typeof container.restart_count === "number" ? container.restart_count : undefined,
      warnings: container.warnings,
      errors: container.errors,
    }));
  } catch {
    return [];
  }
}

async function fetchLiveBrokerPositionTickers(): Promise<string[]> {
  if (!ROBOMARKETS_API_KEY || !ROBOMARKETS_ACCOUNT_ID) return [];

  try {
    const res = await fetch(`${ROBOMARKETS_API_URL}/accounts/${ROBOMARKETS_ACCOUNT_ID}/deals`, {
      cache: "no-store",
      signal: AbortSignal.timeout(8000),
      headers: {
        Authorization: `Bearer ${ROBOMARKETS_API_KEY}`,
        Accept: "application/json",
      },
    });
    if (!res.ok) {
      console.warn("Live broker positions fetch failed", res.status);
      return [];
    }

    const data = await res.json();
    const deals = Array.isArray(data) ? data : (data.deals || data.data);
    if (!Array.isArray(deals)) return [];

    return Array.from(new Set(
      deals
        .filter((deal: Record<string, unknown>) => String(deal.status ?? "open").toLowerCase() === "open")
        .map((deal: Record<string, unknown>) => String(deal.ticker ?? "").split(".")[0].trim().toUpperCase())
        .filter(Boolean)
    ));
  } catch (error) {
    console.warn("Live broker positions fetch failed", error);
    return [];
  }
}

export async function GET() {
  try {
    await ensureSettingsTable();

    const [settingsResult, runResult, candidatesResult, activeResult, liveBrokerTickers, containers] = await Promise.all([
      pool.query(`
        SELECT key, value, value_type, label, description, updated_at
        FROM stock_auto_trade_settings
        WHERE key = ANY($1::text[])
        ORDER BY array_position($1::text[], key)
      `, [Object.keys(SETTING_DEFS)]),
      pool.query(`
        SELECT id, trade_date, status, enabled, dry_run, started_at, validated_at,
               traded_at, completed_at, config, created_at, updated_at
        FROM stock_auto_trade_runs
        ORDER BY trade_date DESC, id DESC
        LIMIT 1
      `),
      pool.query(`
        SELECT id, run_id, trade_date, rank, ticker, status,
               candidate_score::float8 AS candidate_score,
               scanner_name, order_bias, pm_status, pm_direction,
               broker_bid::float8 AS broker_bid,
               broker_ask::float8 AS broker_ask,
               broker_last::float8 AS broker_last,
               broker_spread_pct::float8 AS broker_spread_pct,
               broker_quote_age_minutes,
               relative_volume::float8 AS relative_volume,
               intraday_relative_volume::float8 AS intraday_relative_volume,
               planned_entry::float8 AS planned_entry,
               planned_stop_loss::float8 AS planned_stop_loss,
               planned_take_profit::float8 AS planned_take_profit,
               planned_quantity, robomarkets_order_id, stock_order_id, reason, updated_at
        FROM stock_auto_trade_candidates
        WHERE run_id = (SELECT id FROM stock_auto_trade_runs ORDER BY trade_date DESC, id DESC LIMIT 1)
        ORDER BY rank NULLS LAST, id
        LIMIT 40
      `),
      pool.query(`
        WITH active_tickers AS (
          SELECT UPPER(split_part(ticker, '.', 1)) AS normalized_ticker
          FROM stock_orders
          WHERE status IN ('pending', 'submitted', 'partially_filled')
        )
        SELECT DISTINCT normalized_ticker
        FROM active_tickers
        WHERE normalized_ticker <> ''
      `),
      fetchLiveBrokerPositionTickers(),
      getContainers(),
    ]);

    const settings = settingsResult.rows.map((row) => ({
      key: row.key,
      value: typedValue(row.value, row.value_type),
      raw_value: row.value,
      value_type: row.value_type,
      label: row.label,
      description: row.description,
      updated_at: row.updated_at,
      min: SETTING_DEFS[row.key]?.min,
      max: SETTING_DEFS[row.key]?.max,
    }));

    const watchedContainers = ["stock-auto-trader", "stock-breakeven-monitor"];
    const containerStatus = watchedContainers.map((name) => {
      const found = containers.find((container) => container.name === name);
      return found ?? { name, state: "unknown", status: "unavailable" };
    });

    return NextResponse.json({
      settings,
      latest_run: runResult.rows[0] ?? null,
      candidates: candidatesResult.rows,
      active_count: new Set([
        ...activeResult.rows.map((row) => String(row.normalized_ticker ?? "")),
        ...liveBrokerTickers,
      ].filter(Boolean)).size,
      containers: containerStatus,
    });
  } catch (error) {
    console.error("Failed to load auto trader settings", error);
    return NextResponse.json({ error: "Failed to load auto trader settings" }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object" || typeof body.settings !== "object") {
    return NextResponse.json({ error: "settings object is required" }, { status: 400 });
  }

  try {
    await ensureSettingsTable();
    const entries = Object.entries(body.settings as Record<string, unknown>);
    if (entries.length === 0) {
      return NextResponse.json({ error: "No settings supplied" }, { status: 400 });
    }

    for (const [key, value] of entries) {
      const serialized = parseSetting(key, value);
      const def = SETTING_DEFS[key];
      await pool.query(
        `UPDATE stock_auto_trade_settings
         SET value = $1, value_type = $2, label = $3, description = $4, updated_at = NOW()
         WHERE key = $5`,
        [serialized, def.type, def.label, def.description, key]
      );
    }

    return GET();
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Failed to update settings" }, { status: 400 });
  }
}
