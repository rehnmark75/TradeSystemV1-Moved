import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadGlobal() {
  const result = await strategyConfigPool.query(
    `
      SELECT *
      FROM range_structure_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC, id DESC
      LIMIT 1
    `,
  );
  return result.rows[0] ?? null;
}

export async function GET(_request: Request, { params }: { params: { epic: string } }) {
  try {
    const globalConfig = await loadGlobal();
    if (!globalConfig) {
      return NextResponse.json(
        { error: "No active RANGE_STRUCTURE config found" },
        { status: 404 },
      );
    }

    const overridesResult = await strategyConfigPool.query(
      `
        SELECT *
        FROM range_structure_pair_overrides
        WHERE epic = $1
        LIMIT 1
      `,
      [params.epic],
    );
    const override = overridesResult.rows[0] ?? null;

    const globalEnabled = true;
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? override.is_enabled
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : Boolean(overrideEnabled);

    const monitorOnly =
      override && override.monitor_only !== null && override.monitor_only !== undefined
        ? Boolean(override.monitor_only)
        : true;

    const effective: Record<string, unknown> = { ...globalConfig, strategy_name: "RANGE_STRUCTURE" };
    if (override) {
      const parameterOverrides = override.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);
      Object.entries(override).forEach(([key, value]) => {
        if (
          [
            "id",
            "epic",
            "pair_name",
            "created_at",
            "updated_at",
            "parameter_overrides",
            "notes",
            "disabled_reason",
          ].includes(key)
        ) {
          return;
        }
        if (value !== null && value !== undefined) effective[key] = value;
      });
    }

    return NextResponse.json({
      epic: params.epic,
      global: { ...globalConfig, strategy_name: "RANGE_STRUCTURE" },
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
    console.error("Failed to load effective RANGE_STRUCTURE config", error);
    return NextResponse.json(
      { error: "Failed to load effective RANGE_STRUCTURE config" },
      { status: 500 },
    );
  }
}
