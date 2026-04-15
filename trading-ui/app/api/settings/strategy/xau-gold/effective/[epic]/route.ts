import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

function coerce(value: string, valueType: string) {
  const t = String(valueType ?? "string").toLowerCase();
  if (t === "int" || t === "integer") return parseInt(value, 10);
  if (t === "float") return Number(value);
  if (t === "bool") return value.toLowerCase() === "true";
  if (t === "json") return JSON.parse(value);
  return value;
}

async function loadGlobal(configSet: string) {
  const result = await strategyConfigPool.query(
    `
      SELECT parameter_name, parameter_value, value_type, updated_at, updated_by, change_reason
      FROM xau_gold_global_config
      WHERE is_active = TRUE AND config_set = $1
      ORDER BY display_order, parameter_name
    `,
    [configSet]
  );

  const global: Record<string, unknown> = {
    updated_at: result.rows.reduce(
      (latest: string, row: { updated_at: string }) =>
        !latest || new Date(row.updated_at) > new Date(latest) ? row.updated_at : latest,
      ""
    ),
    updated_by: result.rows.find((row: { updated_by?: string | null }) => row.updated_by)?.updated_by ?? null,
    change_reason:
      result.rows.find((row: { change_reason?: string | null }) => row.change_reason)?.change_reason ?? null,
    config_set: configSet,
  };

  for (const row of result.rows) {
    global[row.parameter_name] = coerce(String(row.parameter_value), String(row.value_type));
  }
  return global;
}

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const globalConfig = await loadGlobal(configSet);

    const overridesResult = await strategyConfigPool.query(
      `
        SELECT *
        FROM xau_gold_pair_overrides
        WHERE config_set = $1 AND epic = $2
        LIMIT 1
      `,
      [configSet, params.epic]
    );
    const override = overridesResult.rows[0] ?? null;

    const enabledPairs = Array.isArray(globalConfig.enabled_pairs) ? globalConfig.enabled_pairs : [];
    const globalEnabled = enabledPairs.includes(params.epic);
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? override.is_enabled
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : Boolean(overrideEnabled);
    const monitorOnly = Boolean(override?.monitor_only);

    const effective = { ...globalConfig };
    if (override) {
      const parameterOverrides = override.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);
      Object.entries(override).forEach(([key, value]) => {
        if (
          ["id", "config_set", "epic", "created_at", "updated_at", "updated_by", "change_reason", "parameter_overrides"].includes(key)
        ) {
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
    console.error("Failed to load effective XAU_GOLD config", error);
    return NextResponse.json(
      { error: "Failed to load effective XAU_GOLD config" },
      { status: 500 }
    );
  }
}
