import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_CONFIG_SET = "demo";

type KVRow = {
  parameter_name: string;
  parameter_value: string;
  value_type: string;
  updated_at: Date;
};

function coerce(value: string, type: string): unknown {
  if (type === "bool") return value.toLowerCase() === "true";
  if (type === "int") return parseInt(value, 10);
  if (type === "float") return parseFloat(value);
  return value;
}

function normalizeTimestamp(value: unknown): string {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString();
}

function pivotRows(rows: KVRow[]): Record<string, unknown> {
  const flat: Record<string, unknown> = {};
  let maxTs = new Date(0);
  for (const row of rows) {
    flat[row.parameter_name] = coerce(row.parameter_value, row.value_type);
    const ts = row.updated_at instanceof Date ? row.updated_at : new Date(String(row.updated_at));
    if (ts > maxTs) maxTs = ts;
  }
  return { ...flat, updated_at: maxTs.toISOString() };
}

async function loadRows(configSet: string, client?: { query: Function }): Promise<KVRow[]> {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `SELECT * FROM fa_or_atr_trail_global_config WHERE config_set = $1 AND is_active = TRUE ORDER BY id`,
    [configSet],
  );
  return result.rows;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const rows = await loadRows(configSet);
    if (!rows.length) {
      return NextResponse.json({ error: `No active FA_OR_ATR_TRAIL config found for config_set='${configSet}'` }, { status: 404 });
    }
    return NextResponse.json({ ...pivotRows(rows), strategy_name: "FA_OR_ATR_TRAIL" });
  } catch (error) {
    console.error("Failed to load FA_OR_ATR_TRAIL config", error);
    return NextResponse.json({ error: "Failed to load FA_OR_ATR_TRAIL config" }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { updates, updated_at, config_set } = body as {
    updates?: Record<string, unknown>;
    updated_at?: string;
    config_set?: string;
  };
  const configSet = config_set ?? DEFAULT_CONFIG_SET;

  if (!updates || typeof updates !== "object") {
    return NextResponse.json({ error: "Missing updates payload" }, { status: 400 });
  }
  if (!updated_at) {
    return NextResponse.json({ error: "updated_at is required" }, { status: 400 });
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const rows = await loadRows(configSet, client);
    if (!rows.length) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: `No active FA_OR_ATR_TRAIL config found for config_set='${configSet}'` }, { status: 404 });
    }

    const maxTs = rows.reduce((max, row) => {
      const ts = row.updated_at instanceof Date ? row.updated_at : new Date(String(row.updated_at));
      return ts > max ? ts : max;
    }, new Date(0));

    if (normalizeTimestamp(maxTs) !== updated_at) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "conflict", message: "Settings were updated by another user", current_config: pivotRows(rows) }, { status: 409 });
    }

    const existingParams = new Set(rows.map((row) => row.parameter_name));
    const keys = Object.keys(updates).filter((key) => existingParams.has(key));
    if (!keys.length) {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "No valid fields to update" }, { status: 400 });
    }

    for (const key of keys) {
      await client.query(
        `
          UPDATE fa_or_atr_trail_global_config
          SET parameter_value = $1, updated_at = NOW()
          WHERE config_set = $2 AND parameter_name = $3 AND is_active = TRUE
        `,
        [String(updates[key]), configSet, key],
      );
    }

    await client.query("COMMIT");
    const updatedRows = await loadRows(configSet);
    return NextResponse.json({ ...pivotRows(updatedRows), strategy_name: "FA_OR_ATR_TRAIL" });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to update FA_OR_ATR_TRAIL config", error);
    return NextResponse.json({ error: "Failed to update FA_OR_ATR_TRAIL config" }, { status: 500 });
  } finally {
    client.release();
  }
}
