import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadGlobal() {
  const result = await strategyConfigPool.query(
    `SELECT * FROM donchian_turtle_global_config WHERE is_active = TRUE ORDER BY updated_at DESC LIMIT 1`,
  );
  return result.rows[0] ?? null;
}

export async function GET(_request: Request, { params }: { params: { epic: string } }) {
  try {
    const globalConfig = await loadGlobal();
    if (!globalConfig) {
      return NextResponse.json({ error: "No active DONCHIAN_TURTLE config found" }, { status: 404 });
    }

    const overridesResult = await strategyConfigPool.query(
      `SELECT * FROM donchian_turtle_pair_overrides WHERE epic = $1 LIMIT 1`,
      [params.epic],
    );
    const override = overridesResult.rows[0] ?? null;

    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? override.is_enabled
        : null;
    const effectiveEnabled = overrideEnabled === null ? false : Boolean(overrideEnabled);

    const monitorOnlyColumn =
      override && Object.prototype.hasOwnProperty.call(override, "monitor_only")
        ? override.monitor_only
        : null;
    const monitorOnlyJsonb = override?.parameter_overrides?.monitor_only ?? null;
    const monitorOnly =
      monitorOnlyColumn !== null ? Boolean(monitorOnlyColumn) : Boolean(monitorOnlyJsonb ?? true);

    // Build effective config: global → JSONB overlay → direct-column overlay
    const effective: Record<string, unknown> = { ...globalConfig };
    if (override) {
      const parameterOverrides = override.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);
      Object.entries(override).forEach(([key, value]) => {
        if (["id", "epic", "pair_name", "created_at", "updated_at", "parameter_overrides", "notes"].includes(key)) return;
        if (value !== null && value !== undefined) effective[key] = value;
      });
    }

    return NextResponse.json({
      epic: params.epic,
      global: globalConfig,
      override,
      effective,
      pair_status: {
        global_enabled: false,
        override_enabled: overrideEnabled,
        effective_enabled: effectiveEnabled,
        monitor_only: monitorOnly,
      },
    });
  } catch (error) {
    console.error("Failed to load effective DONCHIAN_TURTLE config", error);
    return NextResponse.json({ error: "Failed to load effective DONCHIAN_TURTLE config" }, { status: 500 });
  }
}
