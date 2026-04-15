import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const GLOBAL_SYSTEM_FIELDS = new Set([
  "id", "created_at", "updated_at", "updated_by", "change_reason", "is_active", "version"
]);

type SnapshotEnvelope = {
  snapshot_format?: string;
  config_set?: string;
  global_config?: Record<string, unknown>;
  pair_overrides?: Array<Record<string, unknown>>;
};

function isSnapshotEnvelope(value: unknown): value is SnapshotEnvelope {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeSnapshotPayload(
  parameterOverrides: Record<string, unknown>,
  fallbackConfigSet: string | null
) {
  if (isSnapshotEnvelope(parameterOverrides) && parameterOverrides.global_config) {
    return {
      configSet: typeof parameterOverrides.config_set === "string"
        ? parameterOverrides.config_set
        : (fallbackConfigSet ?? "demo"),
      globalConfig: parameterOverrides.global_config,
      pairOverrides: Array.isArray(parameterOverrides.pair_overrides)
        ? parameterOverrides.pair_overrides
        : [],
    };
  }

  return {
    configSet: fallbackConfigSet ?? "demo",
    globalConfig: parameterOverrides,
    pairOverrides: [] as Array<Record<string, unknown>>,
  };
}

function stripGlobalSystemFields(config: Record<string, unknown>) {
  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(config)) {
    if (!GLOBAL_SYSTEM_FIELDS.has(key)) {
      filtered[key] = value;
    }
  }
  return filtered;
}

function buildPairOverrideMap(rows: Array<Record<string, unknown>>) {
  const map = new Map<string, Record<string, unknown>>();
  for (const row of rows) {
    const epic = typeof row.epic === "string" ? row.epic : null;
    if (!epic) continue;
    const filtered: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(row)) {
      if (["id", "config_id", "created_at", "updated_at", "updated_by", "change_reason"].includes(key)) {
        continue;
      }
      filtered[key] = value;
    }
    map.set(epic, filtered);
  }
  return map;
}

function valuesEqual(a: unknown, b: unknown) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

/** GET /api/settings/strategy/smc/snapshots/[id]/compare
 *  Compare snapshot values against current config in the same environment.
 */
export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  const id = Number(params.id);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid snapshot ID" }, { status: 400 });
  }

  try {
    const snapResult = await strategyConfigPool.query(
      `SELECT s.*, c.config_set AS base_config_set
       FROM smc_backtest_snapshots s
       LEFT JOIN smc_simple_global_config c ON c.id = s.base_config_id
       WHERE s.id = $1 AND s.is_active = TRUE`,
      [id]
    );
    const snapshot = snapResult.rows[0];

    if (!snapshot) {
      return NextResponse.json({ error: "Snapshot not found" }, { status: 404 });
    }

    const normalized = normalizeSnapshotPayload(
      snapshot.parameter_overrides as Record<string, unknown>,
      snapshot.base_config_set ?? null
    );

    const configResult = await strategyConfigPool.query(
      `SELECT * FROM smc_simple_global_config
       WHERE is_active = TRUE AND config_set = $1
       ORDER BY updated_at DESC
       LIMIT 1`,
      [normalized.configSet]
    );
    const config = configResult.rows[0];

    if (!config) {
      return NextResponse.json(
        { error: `No active config found for config_set='${normalized.configSet}'` },
        { status: 404 }
      );
    }

    const pairOverridesResult = await strategyConfigPool.query(
      `SELECT *
       FROM smc_simple_pair_overrides
       WHERE config_id = $1
       ORDER BY epic ASC`,
      [config.id]
    );

    const snapshotGlobal = normalized.globalConfig;
    const currentGlobal = stripGlobalSystemFields(config);
    const snapshotPairs = buildPairOverrideMap(normalized.pairOverrides);
    const currentPairs = buildPairOverrideMap(pairOverridesResult.rows);

    const diff: Array<{ field: string; snapshot_value: unknown; current_value: unknown; changed: boolean }> = [];

    const globalKeys = new Set([...Object.keys(snapshotGlobal), ...Object.keys(currentGlobal)]);
    for (const field of globalKeys) {
      const snapshotValue = snapshotGlobal[field];
      const currentValue = currentGlobal[field];
      diff.push({
        field,
        snapshot_value: snapshotValue ?? null,
        current_value: currentValue ?? null,
        changed: !valuesEqual(snapshotValue, currentValue),
      });
    }

    const pairEpics = new Set([...snapshotPairs.keys(), ...currentPairs.keys()]);
    for (const epic of pairEpics) {
      const snapshotPair = snapshotPairs.get(epic) ?? {};
      const currentPair = currentPairs.get(epic) ?? {};
      const pairFields = new Set([...Object.keys(snapshotPair), ...Object.keys(currentPair)]);
      for (const field of pairFields) {
        const snapshotValue = snapshotPair[field];
        const currentValue = currentPair[field];
        diff.push({
          field: `pair:${epic}.${field}`,
          snapshot_value: snapshotValue ?? null,
          current_value: currentValue ?? null,
          changed: !valuesEqual(snapshotValue, currentValue),
        });
      }
    }

    diff.sort((a, b) => {
      if (a.changed !== b.changed) return a.changed ? -1 : 1;
      return a.field.localeCompare(b.field);
    });

    return NextResponse.json({
      snapshot_name: snapshot.snapshot_name,
      snapshot_created_at: snapshot.created_at,
      snapshot_values: {
        config_set: normalized.configSet,
        global_config: snapshotGlobal,
        pair_overrides: Object.fromEntries(snapshotPairs),
      },
      current_values: {
        config_set: normalized.configSet,
        global_config: currentGlobal,
        pair_overrides: Object.fromEntries(currentPairs),
      },
      diff,
      changed_count: diff.filter((entry) => entry.changed).length,
    });
  } catch (error) {
    console.error("Failed to compare snapshot", error);
    return NextResponse.json({ error: "Failed to compare snapshot" }, { status: 500 });
  }
}
