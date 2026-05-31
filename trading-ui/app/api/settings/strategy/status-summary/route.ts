import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

type PairStatus = "active" | "monitor" | "disabled";

type StrategySummary = {
  label: string;
  counts: Record<PairStatus, number>;
  pairs: Array<{
    epic: string;
    status: PairStatus;
    override_count: number;
    inherited: boolean;
  }>;
};

const STATUS_FIELDS = new Set(["is_enabled", "monitor_only"]);

const STRATEGIES = {
  smc: {
    label: "SMC_SIMPLE",
    globalTable: "smc_simple_global_config",
    overrideTable: "smc_simple_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: true,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: false,
  },
  "xau-gold": {
    label: "XAU_GOLD",
    globalTable: "xau_gold_global_config",
    overrideTable: "xau_gold_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: false,
  },
  "mean-reversion": {
    label: "MEAN_REVERSION",
    globalTable: "mean_reversion_global_config",
    overrideTable: "mean_reversion_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    defaultGlobalEnabled: true,
    defaultMonitorOnly: false,
  },
  "range-fade": {
    label: "RANGE_FADE",
    globalTable: "range_fade_global_config",
    overrideTable: "range_fade_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    profileColumn: "profile_name",
    profileValue: "5m",
    defaultGlobalEnabled: true,
    defaultMonitorOnly: false,
  },
  "smc-momentum": {
    label: "SMC_MOMENTUM",
    globalTable: "smc_momentum_global_config",
    overrideTable: "smc_momentum_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: true,
  },
  "impulse-fade": {
    label: "IMPULSE_FADE",
    globalTable: "impulse_fade_global_config",
    overrideTable: "impulse_fade_pair_overrides",
    configSetColumn: undefined,
    overrideConfigJoin: false,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: true,
  },
  "fa-or-atr-trail": {
    label: "FA_OR_ATR_TRAIL",
    globalTable: "fa_or_atr_trail_global_config",
    overrideTable: "fa_or_atr_trail_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: true,
  },
  "donchian-turtle": {
    label: "DONCHIAN_TURTLE",
    globalTable: "donchian_turtle_global_config",
    overrideTable: "donchian_turtle_pair_overrides",
    configSetColumn: undefined,
    overrideConfigJoin: false,
    defaultGlobalEnabled: false,
    defaultMonitorOnly: true,
  },
  kama2: {
    label: "KAMA_V2",
    globalTable: "",
    overrideTable: "kama_v2_pair_overrides",
    configSetColumn: "config_set",
    overrideConfigJoin: false,
    globalDefaults: {
      monitor_only: true,
    },
    defaultGlobalEnabled: false,
    defaultMonitorOnly: true,
  },
} as const;

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function addGlobalPairSources(pairs: Set<string>, globalConfig: Record<string, unknown> | null) {
  if (!globalConfig) return;

  const enabledPairs = globalConfig.enabled_pairs;
  if (Array.isArray(enabledPairs)) {
    enabledPairs.forEach((epic) => {
      if (typeof epic === "string") pairs.add(epic);
    });
  }

  const pairPipValues = globalConfig.pair_pip_values;
  if (pairPipValues && typeof pairPipValues === "object" && !Array.isArray(pairPipValues)) {
    Object.keys(pairPipValues).forEach((epic) => pairs.add(epic));
  }
}

function isKvGlobalTable(tableName: string) {
  return tableName === "smc_momentum_global_config" || tableName === "fa_or_atr_trail_global_config";
}

function countOverrides(row: Record<string, unknown>) {
  let count = 0;
  const parameterOverrides = asRecord(row.parameter_overrides);
  count += Object.keys(parameterOverrides).filter((key) => !STATUS_FIELDS.has(key)).length;

  Object.entries(row).forEach(([key, value]) => {
    if (
      [
        "id",
        "config_id",
        "config_set",
        "profile_name",
        "epic",
        "pair_name",
        "created_at",
        "updated_at",
        "updated_by",
        "change_reason",
        "parameter_overrides",
        "notes",
        "disabled_reason",
      ].includes(key) ||
      STATUS_FIELDS.has(key)
    ) {
      return;
    }
    if (value !== null && value !== undefined) count += 1;
  });
  return count;
}

function hasColumnValue(row: Record<string, unknown> | null, key: string) {
  return row ? Object.prototype.hasOwnProperty.call(row, key) && row[key] !== null && row[key] !== undefined : false;
}

function resolveStatus(
  epic: string,
  globalConfig: Record<string, unknown> | null,
  override: Record<string, unknown> | null,
  defaultGlobalEnabled: boolean,
  defaultMonitorOnly: boolean,
): PairStatus {
  const enabledPairs = Array.isArray(globalConfig?.enabled_pairs) ? globalConfig.enabled_pairs : null;
  const globalEnabled = enabledPairs ? enabledPairs.includes(epic) : defaultGlobalEnabled;
  const overrideEnabled = hasColumnValue(override, "is_enabled") ? Boolean(override?.is_enabled) : null;
  const effectiveEnabled = overrideEnabled === null ? globalEnabled : overrideEnabled;

  if (!effectiveEnabled) return "disabled";

  const parameterOverrides = asRecord(override?.parameter_overrides);
  const monitorOnly = hasColumnValue(override, "monitor_only")
    ? Boolean(override?.monitor_only)
    : parameterOverrides.monitor_only !== null && parameterOverrides.monitor_only !== undefined
      ? Boolean(parameterOverrides.monitor_only)
      : Boolean(globalConfig?.monitor_only ?? defaultMonitorOnly);

  return monitorOnly ? "monitor" : "active";
}

async function loadGlobal(strategy: typeof STRATEGIES[keyof typeof STRATEGIES], configSet: string) {
  if ("globalDefaults" in strategy) {
    return strategy.globalDefaults as Record<string, unknown>;
  }

  const clauses = ["is_active = TRUE"];
  const values: unknown[] = [];
  if (strategy.configSetColumn) {
    values.push(configSet);
    clauses.push(`${strategy.configSetColumn} = $${values.length}`);
  }
  if ("profileColumn" in strategy) {
    values.push(strategy.profileValue);
    clauses.push(`${strategy.profileColumn} = $${values.length}`);
  }

  const result = await strategyConfigPool.query(
    `
      SELECT *
      FROM ${strategy.globalTable}
      WHERE ${clauses.join(" AND ")}
      ORDER BY updated_at DESC, id DESC
      ${isKvGlobalTable(strategy.globalTable) ? "" : "LIMIT 1"}
    `,
    values,
  );
  if (!isKvGlobalTable(strategy.globalTable)) {
    return result.rows[0] ?? null;
  }
  if (!result.rows.length) return null;
  const flat: Record<string, unknown> = {};
  for (const row of result.rows) {
    if (!row.parameter_name) continue;
    let value: unknown = row.parameter_value;
    const valueType = String(row.value_type ?? "string").toLowerCase();
    if (valueType === "bool") value = String(value).toLowerCase() === "true";
    else if (valueType === "int") value = parseInt(String(value), 10);
    else if (valueType === "float") value = parseFloat(String(value));
    flat[row.parameter_name] = value;
  }
  return flat;
}

async function loadOverrides(
  strategy: typeof STRATEGIES[keyof typeof STRATEGIES],
  globalConfig: Record<string, unknown> | null,
  configSet: string,
) {
  const clauses: string[] = [];
  const values: unknown[] = [];
  if (strategy.overrideConfigJoin && globalConfig?.id) {
    values.push(globalConfig.id);
    clauses.push(`config_id = $${values.length}`);
  } else if (strategy.configSetColumn) {
    values.push(configSet);
    clauses.push(`${strategy.configSetColumn} = $${values.length}`);
  }
  if ("profileColumn" in strategy) {
    values.push(strategy.profileValue);
    clauses.push(`${strategy.profileColumn} = $${values.length}`);
  }

  const result = await strategyConfigPool.query(
    `
      SELECT *
      FROM ${strategy.overrideTable}
      ${clauses.length ? `WHERE ${clauses.join(" AND ")}` : ""}
      ORDER BY epic ASC
    `,
    values,
  );
  return result.rows ?? [];
}

async function buildStrategySummary(
  key: keyof typeof STRATEGIES,
  configSet: string,
): Promise<StrategySummary> {
  const strategy = STRATEGIES[key];
  const globalConfig = await loadGlobal(strategy, configSet);
  const overrides = await loadOverrides(strategy, globalConfig, configSet);
  const overrideByEpic = new Map<string, Record<string, unknown>>();
  const pairSet = new Set<string>();

  addGlobalPairSources(pairSet, globalConfig);
  overrides.forEach((row: Record<string, unknown>) => {
    if (typeof row.epic === "string") {
      pairSet.add(row.epic);
      overrideByEpic.set(row.epic, row);
    }
  });

  const pairs = [...pairSet].sort().map((epic) => {
    const override = overrideByEpic.get(epic) ?? null;
    return {
      epic,
      status: resolveStatus(
        epic,
        globalConfig,
        override,
        strategy.defaultGlobalEnabled,
        strategy.defaultMonitorOnly,
      ),
      override_count: override ? countOverrides(override) : 0,
      inherited: !override,
    };
  });

  const counts: Record<PairStatus, number> = { active: 0, monitor: 0, disabled: 0 };
  pairs.forEach((pair) => {
    counts[pair.status] += 1;
  });

  return {
    label: strategy.label,
    counts,
    pairs,
  };
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const configSet = searchParams.get("config_set") ?? "demo";
    const entries = await Promise.all(
      (Object.keys(STRATEGIES) as Array<keyof typeof STRATEGIES>).map(async (key) => [
        key,
        await buildStrategySummary(key, configSet),
      ]),
    );

    return NextResponse.json({ strategies: Object.fromEntries(entries) });
  } catch (error) {
    console.error("Failed to load strategy status summary", error);
    return NextResponse.json(
      { error: "Failed to load strategy status summary" },
      { status: 500 },
    );
  }
}
