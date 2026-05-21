import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

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

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;

    const globalResult = await strategyConfigPool.query(
      `SELECT * FROM fa_or_atr_trail_global_config WHERE config_set = $1 AND is_active = TRUE ORDER BY id`,
      [configSet],
    );
    const globalRows: KVRow[] = globalResult.rows;
    if (!globalRows.length) {
      return NextResponse.json({ error: `No active FA_OR_ATR_TRAIL config found for config_set='${configSet}'` }, { status: 404 });
    }

    const globalConfig = pivotRows(globalRows);
    const overrideResult = await strategyConfigPool.query(
      `SELECT * FROM fa_or_atr_trail_pair_overrides WHERE config_set = $1 AND epic = $2 LIMIT 1`,
      [configSet, params.epic],
    );
    const override = overrideResult.rows[0] ?? null;

    const globalEnabled = Boolean(globalConfig.is_active ?? false);
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? Boolean(override.is_enabled)
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : overrideEnabled;
    const monitorOnly =
      override && override.monitor_only !== null && override.monitor_only !== undefined
        ? Boolean(override.monitor_only)
        : true;

    const effective: Record<string, unknown> = { ...globalConfig };
    if (override) {
      Object.assign(effective, override.parameter_overrides ?? {});
      Object.entries(override).forEach(([key, value]) => {
        if (["id", "config_set", "epic", "pair_name", "created_at", "updated_at", "parameter_overrides", "notes"].includes(key)) {
          return;
        }
        if (value !== null && value !== undefined) effective[key] = value;
      });
    }

    return NextResponse.json({
      epic: params.epic,
      global: globalConfig,
      override,
      effective,
      pair_status: {
        global_enabled: globalEnabled,
        override_enabled: overrideEnabled,
        effective_enabled: effectiveEnabled,
        monitor_only: monitorOnly,
      },
    });
  } catch (error) {
    console.error("Failed to load effective FA_OR_ATR_TRAIL config", error);
    return NextResponse.json({ error: "Failed to load effective FA_OR_ATR_TRAIL config" }, { status: 500 });
  }
}
