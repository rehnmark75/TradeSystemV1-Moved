import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadGlobal(configSet: string) {
  const result = await strategyConfigPool.query(
    `
      SELECT *
      FROM mean_reversion_global_config
      WHERE is_active = TRUE AND config_set = $1
      ORDER BY updated_at DESC
      LIMIT 1
    `,
    [configSet],
  );
  return result.rows[0] ?? null;
}

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const globalConfig = await loadGlobal(configSet);
    if (!globalConfig) {
      return NextResponse.json(
        { error: `No active MEAN_REVERSION config found for config_set='${configSet}'` },
        { status: 404 },
      );
    }

    const overridesResult = await strategyConfigPool.query(
      `
        SELECT *
        FROM mean_reversion_pair_overrides
        WHERE config_set = $1 AND epic = $2
        LIMIT 1
      `,
      [configSet, params.epic],
    );
    const override = overridesResult.rows[0] ?? null;

    const globalEnabled = true; // MEAN_REVERSION has no per-pair gate on global (runs concurrent)
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? override.is_enabled
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : Boolean(overrideEnabled);

    // monitor_only: direct column takes precedence, then JSONB fallback
    const monitorOnlyColumn =
      override && Object.prototype.hasOwnProperty.call(override, "monitor_only")
        ? override.monitor_only
        : null;
    const monitorOnlyJsonb = override?.parameter_overrides?.monitor_only ?? null;
    const monitorOnly =
      monitorOnlyColumn !== null ? Boolean(monitorOnlyColumn) : Boolean(monitorOnlyJsonb);

    // Build effective config: global → JSONB overlay → direct-column overlay
    const effective: Record<string, unknown> = { ...globalConfig };
    if (override) {
      const parameterOverrides = override.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);
      Object.entries(override).forEach(([key, value]) => {
        if (
          [
            "id",
            "config_set",
            "epic",
            "created_at",
            "updated_at",
            "updated_by",
            "change_reason",
            "parameter_overrides",
          ].includes(key)
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
    console.error("Failed to load effective MEAN_REVERSION config", error);
    return NextResponse.json(
      { error: "Failed to load effective MEAN_REVERSION config" },
      { status: 500 },
    );
  }
}
