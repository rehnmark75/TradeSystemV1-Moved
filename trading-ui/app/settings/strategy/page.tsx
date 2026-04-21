"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import CategoryNav, { type CategoryNavItem } from "../../../components/settings/CategoryNav";
import SettingsToolbar, { type SettingsMode } from "../../../components/settings/SettingsToolbar";
import ParameterContextPanel from "../../../components/settings/ParameterContextPanel";
import SettingsField from "../../../components/settings/SettingsField";
import SaveModal from "../../../components/settings/SaveModal";
import SnapshotPanel from "../../../components/settings/SnapshotPanel";
import ConflictModal from "../../../components/settings/ConflictModal";
import { useStrategyConfig } from "../../../hooks/settings/useSmcConfig";
import { useStrategyMetadata } from "../../../hooks/settings/useSmcMetadata";
import { useSettingsSearch } from "../../../hooks/settings/useSettingsSearch";
import { useStrategyPairOverrides } from "../../../hooks/settings/usePairOverrides";
import { apiUrl } from "../../../lib/settings/api";
import { logTelemetry } from "../../../lib/settings/telemetry";
import { useEnvironment } from "../../../lib/environment";
import type { SmcParameterMetadata } from "../../../types/settings";

type EffectivePayload = {
  epic: string;
  global: Record<string, unknown>;
  override: Record<string, unknown> | null;
  effective: Record<string, unknown>;
  pair_status?: {
    global_enabled: boolean;
    override_enabled: boolean | null;
    effective_enabled: boolean;
    monitor_only: boolean;
  };
};

const SYSTEM_FIELDS = new Set([
  "id", "created_at", "updated_at", "updated_by", "change_reason",
  "version", "is_active", "enabled_pairs",
]);

const META_FIELDS = new Set([
  "id", "config_id", "epic", "created_at", "updated_at", "updated_by", "change_reason",
]);

const STATUS_FIELDS = new Set(["is_enabled", "monitor_only"]);

const PREFERRED_ORDER = [
  "Tier 1: 4H Directional Bias",
  "Tier 2: 15m Entry Trigger",
  "Tier 3: 5m Execution",
  "Risk Management",
  "Session Filter",
  "Confidence Scoring",
  "MACD Alignment Filter",
  "Swing Proximity Validation (TIER 4)",
  "Adaptive Cooldown",
  "Scalp Mode (High-Frequency Trading)",
  "Scalp",
  "Scalp Qualification",
  "Alternative Triggers",
  "Enabled Trading Pairs",
  "Other",
];

type StrategyKey = "smc" | "xau-gold" | "mean-reversion" | "range-fade";

const STRATEGIES: Record<StrategyKey, {
  label: string;
  apiBase: string;
  draftKey: string;
  effectiveBase: string;
  columnEndpoint: string;
  snapshotCapable: boolean;
  preferredOrder: string[];
}> = {
  smc: {
    label: "SMC_SIMPLE",
    apiBase: "/api/settings/strategy/smc",
    draftKey: "smc-global-settings-draft",
    effectiveBase: "/api/settings/strategy/smc/effective",
    columnEndpoint: "/api/settings/strategy/smc/pairs/columns",
    snapshotCapable: true,
    preferredOrder: PREFERRED_ORDER,
  },
  "xau-gold": {
    label: "XAU_GOLD",
    apiBase: "/api/settings/strategy/xau-gold",
    draftKey: "xau-gold-global-settings-draft",
    effectiveBase: "/api/settings/strategy/xau-gold/effective",
    columnEndpoint: "/api/settings/strategy/xau-gold/pairs/columns",
    snapshotCapable: false,
    preferredOrder: [
      "general",
      "timeframes",
      "indicators",
      "confidence",
      "confluence",
      "risk",
      "regime",
      "session",
      "structure",
      "limits",
      "filters",
      "pairs",
      "Other",
    ],
  },
  "mean-reversion": {
    label: "MEAN_REVERSION",
    apiBase: "/api/settings/strategy/mean-reversion",
    draftKey: "mean-reversion-global-settings-draft",
    effectiveBase: "/api/settings/strategy/mean-reversion/effective",
    columnEndpoint: "/api/settings/strategy/mean-reversion/pairs/columns",
    snapshotCapable: false,
    preferredOrder: [
      "Hard ADX Gates",
      "Bollinger Bands",
      "RSI",
      "Support / Resistance",
      "Risk Management",
      "Timeframes & Cooldown",
      "Regime & Routing",
      "Legacy Oscillators (archived v0)",
      "Other",
    ],
  },
  "range-fade": {
    label: "RANGE_FADE",
    apiBase: "/api/settings/strategy/range-fade",
    draftKey: "range-fade-global-settings-draft",
    effectiveBase: "/api/settings/strategy/range-fade/effective",
    columnEndpoint: "/api/settings/strategy/range-fade/pairs/columns",
    snapshotCapable: false,
    preferredOrder: [
      "General",
      "Timeframes & Cooldown",
      "HTF Bias",
      "Bollinger Bands",
      "RSI",
      "Range Structure",
      "Volatility Gates",
      "Session",
      "Risk Management",
      "Other",
    ],
  },
};

type PairStatus = "inherit" | "active" | "monitor" | "disabled";
const STATUS_LABELS: Record<PairStatus, string> = {
  inherit: "Inherit",
  active: "Active",
  monitor: "Monitor",
  disabled: "Disabled",
};

function toLabel(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function valuesEqual(a: unknown, b: unknown) {
  if (a === b) return true;
  if (typeof a === "object" && typeof b === "object") {
    return JSON.stringify(a) === JSON.stringify(b);
  }
  return false;
}

function hasKey(obj: Record<string, unknown>, key: string) {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

export default function UnifiedStrategySettings() {
  const { environment } = useEnvironment();
  const [strategy, setStrategy] = useState<StrategyKey>("smc");
  const strategyDef = STRATEGIES[strategy];

  // Mode
  const [mode, setMode] = useState<SettingsMode>("global");

  // Global config
  const {
    effectiveData,
    loading: configLoading,
    error: configError,
    changes,
    updateField,
    saveChanges,
    resetChanges,
    conflict,
    setConflict,
    setChanges,
  } = useStrategyConfig(strategyDef.apiBase, strategyDef.draftKey, environment);

  // Metadata
  const { metadata, loading: metadataLoading } = useStrategyMetadata(`${strategyDef.apiBase}/metadata`);

  // Pair overrides
  const {
    overrides,
    loading: overridesLoading,
    bulkAction,
    deleteOverride,
    saveOverride,
    createOverride,
    reload: reloadOverrides,
  } = useStrategyPairOverrides(strategyDef.apiBase, environment);

  // Pair state
  const [selectedEpic, setSelectedEpic] = useState("");
  const [effective, setEffective] = useState<EffectivePayload | null>(null);
  const [overrideColumns, setOverrideColumns] = useState<string[]>([]);
  const [draftOverrides, setDraftOverrides] = useState<Record<string, unknown>>({});
  const [initialOverrides, setInitialOverrides] = useState<Record<string, unknown>>({});
  const [baseParamOverrides, setBaseParamOverrides] = useState<Record<string, unknown>>({});

  // UI state
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState({
    modifiedOnly: false,
    showAdvanced: false,
    overriddenOnly: false,
    requiresRestartOnly: false,
  });
  const [focusedParam, setFocusedParam] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showSnapshotPanel, setShowSnapshotPanel] = useState(false);
  const [changeReason, setChangeReason] = useState("");
  const [pairSaveLoading, setPairSaveLoading] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Refs for section scrolling
  const sectionRefs = useRef<Map<string, HTMLElement>>(new Map());
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy" });
  }, []);

  // Load override columns
  useEffect(() => {
    fetch(apiUrl(strategyDef.columnEndpoint))
      .then((r) => r.json())
      .then((p) => setOverrideColumns(Array.isArray(p?.columns) ? p.columns : []))
      .catch(() => setOverrideColumns([]));
  }, [strategyDef.columnEndpoint]);

  // Derive all pairs
  const globalConfig = effectiveData as Record<string, unknown> | null;
  const enabledPairs = useMemo(() => {
    const v = globalConfig?.enabled_pairs;
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
  }, [globalConfig]);

  const knownPairs = useMemo(() => {
    const v = globalConfig?.pair_pip_values;
    if (!v || typeof v !== "object" || Array.isArray(v)) return [];
    return Object.keys(v);
  }, [globalConfig]);

  const allPairs = useMemo(() => {
    const pairs = new Set<string>([...enabledPairs, ...knownPairs]);
    overrides.forEach((o) => pairs.add(o.epic));
    return Array.from(pairs).sort();
  }, [enabledPairs, knownPairs, overrides]);

  // Auto-select first pair
  useEffect(() => {
    if (!allPairs.length) {
      setSelectedEpic("");
      return;
    }
    if (!selectedEpic || !allPairs.includes(selectedEpic)) {
      setSelectedEpic(allPairs[0]);
    }
  }, [allPairs, selectedEpic]);

  // Load effective config for selected pair
  useEffect(() => {
    if (!selectedEpic) { setEffective(null); return; }
    const controller = new AbortController();
    fetch(
      apiUrl(`${strategyDef.effectiveBase}/${selectedEpic}?config_set=${encodeURIComponent(environment)}`),
      { signal: controller.signal }
    )
      .then((r) => r.json())
      .then(setEffective)
      .catch(() => {});
    return () => controller.abort();
  }, [selectedEpic, environment, strategyDef.effectiveBase]);

  // Sync draft overrides when effective config changes
  useEffect(() => {
    if (!effective) { setDraftOverrides({}); setInitialOverrides({}); setBaseParamOverrides({}); return; }
    const override = effective.override ?? {};
    const paramOverrides = (override.parameter_overrides as Record<string, unknown>) ?? {};
    const columnSet = new Set(overrideColumns.filter((k) => k !== "parameter_overrides"));
    const columnOverrides: Record<string, unknown> = {};
    Object.keys(override).forEach((key) => {
      if (META_FIELDS.has(key) || key === "parameter_overrides") return;
      if (!columnSet.has(key)) return;
      const value = override[key];
      if (value !== null && value !== undefined) {
        columnOverrides[key] = value;
        return;
      }
      if (STATUS_FIELDS.has(key)) columnOverrides[key] = null;
    });
    const merged = { ...paramOverrides, ...columnOverrides };
    setInitialOverrides(merged);
    setDraftOverrides(merged);
    setBaseParamOverrides(paramOverrides);
  }, [effective, overrideColumns]);

  // Override counts (status fields excluded — they're not parameter overrides)
  const overrideCounts = useMemo(() => {
    const counts = new Map<string, number>();
    overrides.forEach((ov) => {
      let count = 0;
      if (ov.parameter_overrides && typeof ov.parameter_overrides === "object") {
        count += Object.keys(ov.parameter_overrides as Record<string, unknown>)
          .filter((k) => !STATUS_FIELDS.has(k)).length;
      }
      Object.entries(ov).forEach(([k, v]) => {
        if (META_FIELDS.has(k) || k === "parameter_overrides" || STATUS_FIELDS.has(k)) return;
        if (v !== null && v !== undefined) count++;
      });
      counts.set(ov.epic, count);
    });
    return counts;
  }, [overrides]);

  // Dirty pair override keys
  const dirtyKeys = useMemo(() => {
    const keys = new Set<string>();
    const allKeys = new Set([...Object.keys(initialOverrides), ...Object.keys(draftOverrides)]);
    allKeys.forEach((key) => {
      const hasInitial = hasKey(initialOverrides, key);
      const hasDraft = hasKey(draftOverrides, key);
      if (hasInitial !== hasDraft) { keys.add(key); return; }
      if (!valuesEqual(initialOverrides[key], draftOverrides[key])) keys.add(key);
    });
    return keys;
  }, [draftOverrides, initialOverrides]);

  // Build metadata map and search index
  const metadataMap = useMemo(() => {
    const map = new Map<string, SmcParameterMetadata>();
    (metadata ?? []).forEach((item) => map.set(item.parameter_name, item));
    return map;
  }, [metadata]);

  const searchIndex = useMemo(() => {
    const index: Record<string, string> = {};
    (metadata ?? []).forEach((item) => {
      index[item.parameter_name] = [
        item.parameter_name, item.display_name, item.description,
        item.help_text, item.category, item.subcategory,
      ].filter(Boolean).join(" ").toLowerCase();
    });
    return index;
  }, [metadata]);

  // Source data for current mode
  const sourceData = mode === "global"
    ? (globalConfig ?? {})
    : (effective?.global ?? globalConfig ?? {});

  const fieldKeys = Object.keys(sourceData).filter((k) => !SYSTEM_FIELDS.has(k));

  const currentChanges = mode === "global" ? changes : draftOverrides;
  const filteredKeys = useSettingsSearch(fieldKeys, sourceData, {}, query, filters.modifiedOnly, searchIndex);

  // Apply additional filters
  const visibleKeys = useMemo(() => {
    let keys = filteredKeys;
    if (filters.requiresRestartOnly) {
      keys = keys.filter((k) => metadataMap.get(k)?.requires_restart);
    }
    if (mode === "pair" && filters.overriddenOnly) {
      keys = keys.filter((k) => hasKey(draftOverrides, k));
    }
    return keys;
  }, [filteredKeys, filters, metadataMap, mode, draftOverrides]);

  // Build sections
  const sections = useMemo(() => {
    const filteredSet = new Set(visibleKeys);
    const sectionsMap = new Map<string, { category: string; items: SmcParameterMetadata[]; subgroups: Map<string, SmcParameterMetadata[]> }>();

    const addItem = (item: SmcParameterMetadata) => {
      const category = item.category || "Other";
      const section = sectionsMap.get(category) ?? { category, items: [] as SmcParameterMetadata[], subgroups: new Map<string, SmcParameterMetadata[]>() };
      const subcategory = item.subcategory?.trim();
      if (subcategory) {
        const bucket = section.subgroups.get(subcategory) ?? [];
        bucket.push(item);
        section.subgroups.set(subcategory, bucket);
      } else {
        section.items.push(item);
      }
      sectionsMap.set(category, section);
    };

    (metadata ?? []).forEach((item) => {
      if (!filteredSet.has(item.parameter_name)) return;
      if (!filters.showAdvanced && item.is_advanced) return;
      addItem(item);
    });

    filteredSet.forEach((field) => {
      if (metadataMap.has(field)) return;
      addItem({ id: 0, parameter_name: field, display_name: toLabel(field), category: "Other", data_type: typeof sourceData[field], display_order: 0 });
    });

    const sortItems = (items: SmcParameterMetadata[]) =>
      [...items].sort((a, b) => ((a.display_order ?? 0) - (b.display_order ?? 0)) || a.display_name.localeCompare(b.display_name));

    return [...sectionsMap.values()]
      .sort((a, b) => {
        const ia = strategyDef.preferredOrder.indexOf(a.category);
        const ib = strategyDef.preferredOrder.indexOf(b.category);
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      })
      .map((s) => ({
        ...s,
        items: sortItems(s.items),
        subgroups: new Map([...s.subgroups.entries()].map(([k, v]) => [k, sortItems(v)])),
      }));
  }, [visibleKeys, metadata, metadataMap, filters.showAdvanced, sourceData, strategyDef.preferredOrder]);

  // Category nav items
  const categoryNavItems: CategoryNavItem[] = useMemo(() => {
    return sections.map((s) => {
      const allFields = [
        ...s.items.map((i) => i.parameter_name),
        ...[...s.subgroups.values()].flat().map((i) => i.parameter_name),
      ];
      const modifiedCount = mode === "global"
        ? allFields.filter((f) => hasKey(changes, f)).length
        : allFields.filter((f) => hasKey(draftOverrides, f)).length;
      const overriddenCount = mode === "pair"
        ? allFields.filter((f) => hasKey(draftOverrides, f)).length
        : 0;
      return { category: s.category, fieldCount: allFields.length, modifiedCount, overriddenCount };
    });
  }, [sections, mode, changes, draftOverrides]);

  // Trading status derived from draftOverrides
  const currentStatus = useMemo((): PairStatus => {
    if (draftOverrides.monitor_only === true || draftOverrides.monitor_only === "true") return "monitor";
    if (draftOverrides.is_enabled === false) return "disabled";
    if (draftOverrides.is_enabled === true) return "active";
    if (draftOverrides.is_enabled === null) return "inherit";
    if (!hasKey(draftOverrides, "is_enabled")) return "inherit";
    return "active";
  }, [draftOverrides]);

  const setStatus = (status: PairStatus) => {
    setDraftOverrides((prev) => {
      const copy = { ...prev };
      if (status === "inherit") {
        delete copy.is_enabled;
        delete copy.monitor_only;
      } else if (status === "disabled") {
        copy.is_enabled = false;
        delete copy.monitor_only;
      } else if (status === "monitor") {
        copy.is_enabled = true;
        copy.monitor_only = true;
      } else {
        copy.is_enabled = true;
        delete copy.monitor_only;
      }
      return copy;
    });
  };

  // Pending count
  const pendingCount = mode === "global" ? Object.keys(changes).length : dirtyKeys.size;

  // Per-pair effective values for context panel
  const perPairEffective = useMemo(() => {
    const map = new Map<string, unknown>();
    if (focusedParam && effective) {
      map.set(selectedEpic, effective.effective[focusedParam]);
    }
    return map;
  }, [focusedParam, effective, selectedEpic]);

  // IntersectionObserver for active category tracking
  useEffect(() => {
    const refs = sectionRefs.current;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveCategory(entry.target.getAttribute("data-category"));
            break;
          }
        }
      },
      { threshold: 0.1, rootMargin: "-100px 0px -60% 0px" }
    );
    refs.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sections]);

  // Scroll to category
  const scrollToCategory = (category: string) => {
    const el = sectionRefs.current.get(category);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // Save handlers
  const handleSave = () => {
    setSaveError(null);
    setShowSaveModal(true);
  };

  const handleConfirmSave = async (reason: string) => {
    setSaveError(null);

    if (mode === "global") {
      const result = await saveChanges({ updatedBy: "admin", changeReason: reason });
      if (!result.success) {
        setSaveError("Failed to save global settings. Check the latest error on the page and try again.");
        return;
      }
      setShowSaveModal(false);
      return;
    }

    if (!selectedEpic) return;
    const columnSet = new Set(overrideColumns.filter((k) => k !== "parameter_overrides"));
    const nextParamOverrides = { ...baseParamOverrides };
    const updates: Record<string, unknown> = {};
    let paramChanged = false;

    dirtyKeys.forEach((key) => {
      const hasDraft = hasKey(draftOverrides, key);
      const nextValue = hasDraft ? draftOverrides[key] : null;
      if (columnSet.has(key)) {
        updates[key] = nextValue;
      } else {
        if (hasDraft) nextParamOverrides[key] = nextValue;
        else delete nextParamOverrides[key];
        paramChanged = true;
      }
    });
    if (paramChanged) updates.parameter_overrides = nextParamOverrides;

    setPairSaveLoading(true);
    try {
      const activePairOverride = overrides.find((o) => o.epic === selectedEpic);
      if (activePairOverride && effective?.override?.updated_at) {
        await saveOverride(selectedEpic, updates, { updatedBy: "admin", changeReason: reason, updatedAt: String(effective.override.updated_at) });
      } else {
        await createOverride(selectedEpic, updates, { updatedBy: "admin", changeReason: reason });
      }
      await reloadOverrides();
      const refreshed = await fetch(apiUrl(`${strategyDef.effectiveBase}/${selectedEpic}?config_set=${encodeURIComponent(environment)}`)).then((r) => r.json());
      setEffective(refreshed);
      setShowSaveModal(false);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Failed to save pair override.");
      return;
    } finally {
      setPairSaveLoading(false);
    }
  };

  const handleDiscard = () => {
    if (mode === "global") resetChanges();
    else { setDraftOverrides(initialOverrides); }
  };

  // Override toggle / update
  const toggleOverride = (field: string, next: boolean) => {
    setDraftOverrides((prev) => {
      const copy = { ...prev };
      if (next) copy[field] = effective?.global[field] ?? "";
      else delete copy[field];
      return copy;
    });
  };

  const updateOverrideValue = (field: string, value: unknown) => {
    setDraftOverrides((prev) => ({ ...prev, [field]: value }));
  };

  // Render a field
  const renderField = (item: SmcParameterMetadata) => {
    const name = item.parameter_name;
    const label = item.display_name || toLabel(name);
    const descParts = [item.help_text ?? item.description];
    if (item.requires_restart) descParts.push("Requires restart");
    const description = descParts.filter(Boolean).join(" · ");

    if (mode === "global") {
      return (
        <SettingsField
          key={name}
          name={name}
          label={label}
          description={description}
          unit={item.unit ?? undefined}
          value={sourceData[name]}
          defaultValue={item.default_value ?? undefined}
          metadata={item}
          pending={hasKey(changes, name)}
          onFocus={() => setFocusedParam(name)}
          onChange={(v) => updateField(name, v)}
        />
      );
    }

    // Pair override mode
    const isOverridden = hasKey(draftOverrides, name);
    return (
      <SettingsField
        key={name}
        name={name}
        label={label}
        description={description}
        unit={item.unit ?? undefined}
        value={isOverridden ? draftOverrides[name] : (effective?.global[name] ?? sourceData[name])}
        defaultValue={item.default_value ?? undefined}
        metadata={item}
        pending={dirtyKeys.has(name)}
        overrideMode
        globalValue={effective?.global[name] ?? sourceData[name]}
        effectiveValue={effective?.effective[name]}
        isOverridden={isOverridden}
        onToggle={(next) => toggleOverride(name, next)}
        onFocus={() => setFocusedParam(name)}
        onChange={(v) => updateOverrideValue(name, v)}
      />
    );
  };

  const isLoading = configLoading || metadataLoading || (mode === "pair" && overridesLoading);

  // Original values for save modal diff
  const originalValues = useMemo(() => {
    if (mode === "global") return globalConfig ?? {};
    return initialOverrides;
  }, [mode, globalConfig, initialOverrides]);

  return (
    <div className="strategy-page">
      <div className="settings-hero">
        <div className="mission-kicker">Strategy Governance</div>
        <h1>Strategy Settings</h1>
        <p>
          Control global and pair-specific strategy behavior with explicit context for overrides, effective values,
          and change risk across environments.
        </p>
      </div>

      <SettingsToolbar
        mode={mode}
        onModeChange={setMode}
        pairs={allPairs}
        selectedPair={selectedEpic}
        pairOverrideCounts={overrideCounts}
        onPairChange={setSelectedEpic}
        query={query}
        onQueryChange={setQuery}
        filters={filters}
        onFiltersChange={setFilters}
        pendingCount={pendingCount}
        onSave={handleSave}
        onSnapshot={() => setShowSnapshotPanel(true)}
        onDiscard={handleDiscard}
        showSnapshotAction={strategyDef.snapshotCapable}
      />

      <div className="settings-segment" style={{ marginBottom: 16 }}>
        {(["smc", "xau-gold", "mean-reversion", "range-fade"] as const).map((key) => (
          <button
            key={key}
            type="button"
            className={`pair-status-btn strategy-selector-btn ${strategy === key ? "selected" : ""}`}
            onClick={() => setStrategy(key)}
          >
            {STRATEGIES[key].label}
          </button>
        ))}
      </div>

      {mode === "pair" && selectedEpic ? (
        <div className="pair-status-bar">
          <span className="pair-status-bar-label">Trading Status</span>
          <div className="pair-status-control">
            {(["inherit", "active", "monitor", "disabled"] as const).map((s) => (
              <button
                key={s}
                className={`pair-status-btn pair-status-btn-${s}${currentStatus === s ? " selected" : ""}`}
                onClick={() => setStatus(s)}
              >
                {STATUS_LABELS[s]}
              </button>
            ))}
          </div>
          <span className="pair-status-bar-hint">
            {currentStatus === "inherit"
              ? `Using global pair status (${effective?.pair_status?.effective_enabled ? "enabled" : "disabled"})`
              : currentStatus === "monitor"
              ? "Signals logged but not traded"
              : currentStatus === "disabled"
              ? "Pair fully disabled"
              : "Pair actively traded"}
          </span>
        </div>
      ) : null}

      <div className="strategy-layout">
        <CategoryNav
          items={categoryNavItems}
          activeCategory={activeCategory ?? undefined}
          onSelect={scrollToCategory}
          showSnapshots={strategyDef.snapshotCapable}
          onSnapshotsClick={() => setShowSnapshotPanel(true)}
        />

        <div className="strategy-content" ref={contentRef}>
          {isLoading ? (
            <div className="settings-placeholder">Loading…</div>
          ) : configError ? (
            <div className="settings-placeholder">Error: {configError}</div>
          ) : (
            sections.map((section) => {
              const allFields = [
                ...section.items.map((i) => i.parameter_name),
                ...[...section.subgroups.values()].flat().map((i) => i.parameter_name),
              ];
              return (
                <section
                  key={section.category}
                  className="strategy-section"
                  data-category={section.category}
                  ref={(el) => {
                    if (el) sectionRefs.current.set(section.category, el);
                    else sectionRefs.current.delete(section.category);
                  }}
                >
                  <div className="strategy-section-header">
                    <h2>{section.category}</h2>
                    <span className="strategy-section-count">{allFields.length} fields</span>
                  </div>

                  {section.subgroups.size > 0
                    ? [...section.subgroups.entries()].map(([title, items]) => (
                        <div key={title} className="strategy-subgroup">
                          <div className="strategy-subgroup-title">{title}</div>
                          <div className="strategy-field-grid">
                            {items.map(renderField)}
                          </div>
                        </div>
                      ))
                    : (
                      <div className="strategy-field-grid">
                        {section.items.map(renderField)}
                      </div>
                    )
                  }
                </section>
              );
            })
          )}
        </div>

        <ParameterContextPanel
          paramName={focusedParam}
          metadata={focusedParam ? metadataMap.get(focusedParam) : undefined}
          currentValue={focusedParam ? (sourceData[focusedParam]) : undefined}
          allPairEpics={allPairs}
          allPairEffectiveValues={perPairEffective}
          onReset={focusedParam && mode === "global"
            ? () => {
                const meta = metadataMap.get(focusedParam!);
                if (meta?.default_value !== undefined) {
                  updateField(focusedParam!, meta.default_value);
                }
              }
            : undefined}
        />
      </div>

      {showSaveModal ? (
        <SaveModal
          changes={mode === "global" ? (changes as Record<string, unknown>) : Object.fromEntries([...dirtyKeys].map((k) => [k, draftOverrides[k]]))}
          originalValues={originalValues as Record<string, unknown>}
          onConfirm={handleConfirmSave}
          onCancel={() => {
            if (pairSaveLoading) return;
            setSaveError(null);
            setShowSaveModal(false);
          }}
          saving={pairSaveLoading}
          error={saveError}
        />
      ) : null}

      {showSnapshotPanel && strategyDef.snapshotCapable ? (
        <SnapshotPanel
          configSet={environment}
          onClose={() => setShowSnapshotPanel(false)}
          onRestored={() => window.location.reload()}
        />
      ) : null}

      <ConflictModal
        open={!!conflict}
        current={conflict}
        pending={changes}
        onClose={() => setConflict(null)}
        onResolve={async ({ action, mergedChanges }) => {
          if (!conflict) return;
          if (action === "discard") { resetChanges(); setConflict(null); return; }
          if (mergedChanges) {
            setChanges(mergedChanges);
            await saveChanges({ updatedBy: "admin", changeReason: changeReason || "Conflict resolution" }, mergedChanges, (conflict as any).updated_at);
            setConflict(null);
          }
        }}
      />
    </div>
  );
}
