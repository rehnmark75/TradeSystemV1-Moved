import { NextResponse } from "next/server";
import {
  DEFAULT_CONFIG_SET,
  INSIDE_DAY_GLOBAL_DEFAULTS,
  OVERRIDE_TABLE,
  applyOverride,
  strategyConfigPool,
} from "../../common";

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const globalConfig = { ...INSIDE_DAY_GLOBAL_DEFAULTS };

    const overrideResult = await strategyConfigPool.query(
      `SELECT * FROM ${OVERRIDE_TABLE} WHERE config_set = $1 AND epic = $2 LIMIT 1`,
      [configSet, params.epic],
    );
    const override = overrideResult.rows[0] ?? null;
    const effective = applyOverride(globalConfig, override);

    const enabledPairs = Array.isArray(globalConfig.enabled_pairs) ? globalConfig.enabled_pairs : [];
    const globalEnabled = enabledPairs.includes(params.epic);
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? Boolean(override.is_enabled)
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : overrideEnabled;
    const monitorOnly =
      override && override.monitor_only !== null && override.monitor_only !== undefined
        ? Boolean(override.monitor_only)
        : Boolean(globalConfig.monitor_only);

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
    console.error("Failed to load effective INSIDE_DAY config", error);
    return NextResponse.json({ error: "Failed to load effective INSIDE_DAY config" }, { status: 500 });
  }
}
