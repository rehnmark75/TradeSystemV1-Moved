import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadActiveSmcConfig(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT *
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `
  );
  return result.rows[0] ?? null;
}

export async function GET(
  _request: Request,
  { params }: { params: { epic: string } }
) {
  try {
    const globalConfig = await loadActiveSmcConfig();
    if (!globalConfig) {
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const overridesResult = await strategyConfigPool.query(
      `
        SELECT *
        FROM smc_simple_pair_overrides
        WHERE config_id = $1 AND epic = $2
        LIMIT 1
      `,
      [globalConfig.id, params.epic]
    );
    const override = overridesResult.rows[0] ?? null;

    const effective = { ...globalConfig };
    if (override) {
      const overrideFields = { ...override };
      delete overrideFields.id;
      delete overrideFields.config_id;
      delete overrideFields.epic;
      delete overrideFields.created_at;
      delete overrideFields.updated_at;
      delete overrideFields.updated_by;
      delete overrideFields.change_reason;

      const parameterOverrides = overrideFields.parameter_overrides ?? {};
      Object.assign(effective, parameterOverrides);

      Object.entries(overrideFields).forEach(([key, value]) => {
        if (key === "parameter_overrides") return;
        if (value !== null && value !== undefined) {
          effective[key] = value;
        }
      });
    }

    return NextResponse.json({
      epic: params.epic,
      global: globalConfig,
      override,
      effective
    });
  } catch (error) {
    console.error("Failed to load effective SMC config", error);
    return NextResponse.json(
      { error: "Failed to load effective SMC config" },
      { status: 500 }
    );
  }
}
