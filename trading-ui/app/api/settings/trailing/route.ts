import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const CONFIG_FIELDS = [
  "early_breakeven_trigger_points",
  "early_breakeven_buffer_points",
  "stage1_trigger_points",
  "stage1_lock_points",
  "stage2_trigger_points",
  "stage2_lock_points",
  "stage3_trigger_points",
  "stage3_atr_multiplier",
  "stage3_min_distance",
  "min_trail_distance",
  "break_even_trigger_points",
  "enable_partial_close",
  "partial_close_trigger_points",
  "partial_close_size",
] as const;

const ALLOWED_STRATEGIES = new Set([
  "DEFAULT",
  "SMC_SIMPLE",
  "XAU_GOLD",
  "RANGE_FADE",
  "MEAN_REVERSION",
  "RANGE_STRUCTURE",
]);

async function invalidateTrailingCache(configSet: string) {
  const baseUrl =
    configSet === "live"
      ? process.env.FASTAPI_LIVE_URL || "http://fastapi-live:8000"
      : process.env.FASTAPI_DEV_URL || "http://fastapi-dev:8000";

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1500);
  try {
    await fetch(`${baseUrl}/api/trailing-config/invalidate-cache`, {
      method: "POST",
      headers: { "x-apim-gateway": "verified" },
      signal: controller.signal,
    });
  } catch (error) {
    console.warn("Failed to invalidate trailing config cache", error);
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeStrategy(s: string | null | undefined): string {
  const u = (s ?? "DEFAULT").toUpperCase();
  return ALLOWED_STRATEGIES.has(u) ? u : "DEFAULT";
}

function isOverrideValue(value: unknown) {
  return value !== null && value !== undefined;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const configSet = searchParams.get("config_set") ?? "demo";
  const isScalpParam = searchParams.get("is_scalp");
  const isScalp = isScalpParam === "true";
  const strategyParam = searchParams.get("strategy");

  try {
    const cols = [
      "id",
      "strategy",
      "config_set",
      "epic",
      "is_scalp",
      "is_active",
      ...CONFIG_FIELDS,
      "updated_by",
      "change_reason",
      "updated_at",
    ].join(", ");

    const requestedStrategy = strategyParam
      ? normalizeStrategy(strategyParam)
      : "DEFAULT";

    // Fetch DEFAULT rows + the requested strategy's own rows; layer strategy
    // over DEFAULT per (epic, is_scalp) so non-DEFAULT views render the same
    // pairs as DEFAULT, with strategy-specific overrides applied. Rows that
    // come purely from DEFAULT are marked `inherited: true` so the editor
    // can present them as overridable templates.
    const wantedStrategies = ["DEFAULT", requestedStrategy];
    const whereClauses: string[] = [
      "config_set = $1",
      "is_active = TRUE",
      "strategy = ANY($2::text[])",
    ];
    const params: unknown[] = [configSet, wantedStrategies];
    if (isScalpParam !== null) {
      whereClauses.push(`is_scalp = $${params.length + 1}`);
      params.push(isScalp);
    }

    const result = await strategyConfigPool.query(
      `
        SELECT ${cols}
        FROM trailing_pair_config
        WHERE ${whereClauses.join(" AND ")}
        ORDER BY epic = 'DEFAULT' DESC, is_scalp ASC, epic ASC
      `,
      params
    );

    type Row = Record<string, unknown> & {
      strategy: string;
      epic: string;
      is_scalp: boolean;
    };
    const allRows = (result.rows ?? []) as Row[];
    if (requestedStrategy === "DEFAULT") {
      return NextResponse.json({
        rows: allRows,
        strategies: Array.from(ALLOWED_STRATEGIES).sort(),
      });
    }

    // Group by (epic, is_scalp). For strategy rows, return the effective
    // runtime view: DEFAULT fields layered under non-null strategy overrides.
    const byKey = new Map<string, { def?: Row; spec?: Row }>();
    for (const r of allRows) {
      const key = `${r.epic}::${r.is_scalp}`;
      const slot = byKey.get(key) ?? {};
      if (r.strategy === requestedStrategy) slot.spec = r;
      else if (r.strategy === "DEFAULT") slot.def = r;
      byKey.set(key, slot);
    }
    const merged: Row[] = [];
    for (const { def, spec } of byKey.values()) {
      if (spec) {
        const effective: Row = {
          ...(def ?? {}),
          ...spec,
          inherited: false,
          override_field_count: CONFIG_FIELDS.filter((field) =>
            isOverrideValue(spec[field])
          ).length,
        };
        for (const field of CONFIG_FIELDS) {
          effective[field] = isOverrideValue(spec[field])
            ? spec[field]
            : def?.[field] ?? null;
        }
        merged.push(effective);
      } else if (def) {
        merged.push({
          ...def,
          id: null,
          strategy: requestedStrategy,
          inherited: true,
          override_field_count: 0,
        });
      }
    }
    merged.sort((a, b) => {
      if (a.epic === "DEFAULT" && b.epic !== "DEFAULT") return -1;
      if (b.epic === "DEFAULT" && a.epic !== "DEFAULT") return 1;
      if (a.is_scalp !== b.is_scalp) return a.is_scalp ? 1 : -1;
      return a.epic.localeCompare(b.epic);
    });
    return NextResponse.json({
      rows: merged,
      strategies: Array.from(ALLOWED_STRATEGIES).sort(),
    });
  } catch (error) {
    console.error("Failed to load trailing configs", error);
    return NextResponse.json(
      { error: "Failed to load trailing configs" },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const {
    strategy,
    epic,
    is_scalp,
    updates,
    updated_by,
    change_reason,
    config_set,
  } = body as {
    strategy?: string;
    epic?: string;
    is_scalp?: boolean;
    updates?: Record<string, unknown>;
    updated_by?: string;
    change_reason?: string;
    config_set?: string;
  };

  const configSet = config_set ?? "demo";
  const strategyName = normalizeStrategy(strategy);
  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }
  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  const allowed = new Set<string>(CONFIG_FIELDS);
  const keys = Object.keys(updates ?? {}).filter((k) => allowed.has(k));
  if (keys.length === 0) {
    return NextResponse.json(
      { error: "No valid fields provided" },
      { status: 400 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");

    const insertCols = ["strategy", "config_set", "epic", "is_scalp", ...keys, "updated_by", "change_reason"];
    const values: unknown[] = [strategyName, configSet, epic, Boolean(is_scalp)];
    keys.forEach((k) => values.push((updates as Record<string, unknown>)[k]));
    values.push(updated_by, change_reason);

    const placeholders = insertCols.map((_, i) => `$${i + 1}`).join(", ");
    const setClause = keys
      .map((k) => `${k} = EXCLUDED.${k}`)
      .concat([
        "updated_by = EXCLUDED.updated_by",
        "change_reason = EXCLUDED.change_reason",
        "updated_at = NOW()",
      ])
      .join(", ");

    const result = await client.query(
      `
        INSERT INTO trailing_pair_config (${insertCols.join(", ")})
        VALUES (${placeholders})
        ON CONFLICT (strategy, config_set, epic, is_scalp)
        DO UPDATE SET ${setClause}
        RETURNING *
      `,
      values
    );

    await client.query(
      `
        INSERT INTO trailing_config_audit
          (config_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, 'UPSERT', $2, $3, $4, $5)
      `,
      [
        result.rows[0]?.id ?? null,
        updated_by,
        change_reason,
        null,
        JSON.stringify(updates),
      ]
    );

    await client.query("COMMIT");
    await invalidateTrailingCache(configSet);
    return NextResponse.json(result.rows[0]);
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to upsert trailing config", error);
    return NextResponse.json(
      { error: "Failed to upsert trailing config" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
