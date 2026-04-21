import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_PROFILE = "5m";
const DEFAULT_CONFIG_SET = "demo";

async function loadGlobal(profileName: string, configSet: string) {
  const result = await strategyConfigPool.query(
    `
      SELECT *
      FROM eurusd_range_fade_global_config
      WHERE is_active = TRUE AND profile_name = $1 AND config_set = $2
      ORDER BY updated_at DESC
      LIMIT 1
    `,
    [profileName, configSet],
  );
  return result.rows[0] ?? null;
}

export async function GET(request: Request, { params }: { params: { epic: string } }) {
  try {
    const { searchParams } = new URL(request.url);
    const profileParam = searchParams.get("profile_name");
    const profile = profileParam && /^(5m|15m)$/.test(profileParam) ? profileParam : DEFAULT_PROFILE;
    const configSet = searchParams.get("config_set") ?? DEFAULT_CONFIG_SET;
    const globalConfig = await loadGlobal(profile, configSet);
    if (!globalConfig) {
      return NextResponse.json(
        { error: `No active RANGE_FADE config found for profile_name='${profile}', config_set='${configSet}'` },
        { status: 404 },
      );
    }

    const overridesResult = await strategyConfigPool.query(
      `
        SELECT *
        FROM eurusd_range_fade_pair_overrides
        WHERE profile_name = $1 AND config_set = $2 AND epic = $3
        LIMIT 1
      `,
      [profile, configSet, params.epic],
    );
    const override = overridesResult.rows[0] ?? null;

    const globalEnabled = true;
    const overrideEnabled =
      override && Object.prototype.hasOwnProperty.call(override, "is_enabled")
        ? override.is_enabled
        : null;
    const effectiveEnabled = overrideEnabled === null ? globalEnabled : Boolean(overrideEnabled);

    const monitorOnlyColumn =
      override && Object.prototype.hasOwnProperty.call(override, "monitor_only")
        ? override.monitor_only
        : null;
    const monitorOnlyJsonb = override?.parameter_overrides?.monitor_only ?? null;
    const monitorOnly =
      monitorOnlyColumn !== null ? Boolean(monitorOnlyColumn) : Boolean(monitorOnlyJsonb ?? globalConfig.monitor_only);

    const effective: Record<string, unknown> = { ...globalConfig };
    if (override) {
      const parameterOverrides = override.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);
      Object.entries(override).forEach(([key, value]) => {
        if (
          [
            "id",
            "profile_name",
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
    console.error("Failed to load effective RANGE_FADE config", error);
    return NextResponse.json(
      { error: "Failed to load effective RANGE_FADE config" },
      { status: 500 },
    );
  }
}
