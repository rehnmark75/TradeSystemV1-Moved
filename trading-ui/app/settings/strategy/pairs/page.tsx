"use client";

import { useEffect, useMemo, useState } from "react";
import BulkActionBar from "../../../../components/settings/BulkActionBar";
import SettingsSearch from "../../../../components/settings/SettingsSearch";
import OverrideField from "../../../../components/settings/OverrideField";
import { usePairOverrides } from "../../../../hooks/settings/usePairOverrides";
import { useSmcMetadata } from "../../../../hooks/settings/useSmcMetadata";
import { useSettingsSearch } from "../../../../hooks/settings/useSettingsSearch";
import { apiUrl } from "../../../../lib/settings/api";
import { logTelemetry } from "../../../../lib/settings/telemetry";
import type { SmcParameterMetadata } from "../../../../types/settings";

type EffectivePayload = {
  epic: string;
  global: Record<string, unknown>;
  override: Record<string, unknown> | null;
  effective: Record<string, unknown>;
};

const SYSTEM_FIELDS = new Set([
  "id",
  "created_at",
  "updated_at",
  "updated_by",
  "change_reason",
  "version",
  "is_active",
  "enabled_pairs"
]);

const META_FIELDS = new Set([
  "id",
  "config_id",
  "epic",
  "created_at",
  "updated_at",
  "updated_by",
  "change_reason"
]);

function toLabel(value: string) {
  return value.replace(/_/g, " ");
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

export default function SmcPairOverridesPage() {
  const {
    overrides,
    loading,
    error,
    bulkAction,
    deleteOverride,
    saveOverride,
    createOverride,
    reload
  } = usePairOverrides();
  const { metadata, loading: metadataLoading } = useSmcMetadata();
  const [selected, setSelected] = useState<string[]>([]);
  const [pairQuery, setPairQuery] = useState("");
  const [fieldQuery, setFieldQuery] = useState("");
  const [showPairOverridesOnly, setShowPairOverridesOnly] = useState(false);
  const [showFieldOverridesOnly, setShowFieldOverridesOnly] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [updatedBy, setUpdatedBy] = useState("");
  const [changeReason, setChangeReason] = useState("");
  const [selectedEpic, setSelectedEpic] = useState("");
  const [effective, setEffective] = useState<EffectivePayload | null>(null);
  const [effectiveLoading, setEffectiveLoading] = useState(false);
  const [effectiveError, setEffectiveError] = useState<string | null>(null);
  const [globalConfig, setGlobalConfig] = useState<Record<string, unknown> | null>(null);
  const [overrideColumns, setOverrideColumns] = useState<string[]>([]);
  const [draftOverrides, setDraftOverrides] = useState<Record<string, unknown>>({});
  const [initialOverrides, setInitialOverrides] = useState<Record<string, unknown>>({});
  const [baseParamOverrides, setBaseParamOverrides] = useState<Record<string, unknown>>({});

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy/pairs" });
  }, []);

  useEffect(() => {
    fetch(apiUrl("/api/settings/strategy/smc"))
      .then((res) => res.json())
      .then((payload) => setGlobalConfig(payload))
      .catch(() => setGlobalConfig(null));
  }, []);

  useEffect(() => {
    fetch(apiUrl("/api/settings/strategy/smc/pairs/columns"))
      .then((res) => res.json())
      .then((payload) =>
        setOverrideColumns(Array.isArray(payload?.columns) ? payload.columns : [])
      )
      .catch(() => setOverrideColumns([]));
  }, []);

  const enabledPairs = useMemo(() => {
    const value = globalConfig?.enabled_pairs;
    return Array.isArray(value) ? value.filter((item) => typeof item === "string") : [];
  }, [globalConfig]);

  const allPairs = useMemo(() => {
    const pairs = new Set<string>(enabledPairs);
    overrides.forEach((item) => pairs.add(item.epic));
    return Array.from(pairs).sort();
  }, [enabledPairs, overrides]);

  useEffect(() => {
    if (!selectedEpic && allPairs.length) {
      setSelectedEpic(allPairs[0]);
    }
  }, [allPairs, selectedEpic]);

  useEffect(() => {
    if (!selectedEpic) {
      setEffective(null);
      return;
    }
    const controller = new AbortController();
    setEffectiveLoading(true);
    setEffectiveError(null);
    fetch(apiUrl(`/api/settings/strategy/smc/effective/${selectedEpic}`), {
      signal: controller.signal
    })
      .then(async (res) => {
        if (!res.ok) {
          const payload = await res.json().catch(() => null);
          throw new Error(payload?.error ?? "Failed to load effective config");
        }
        return res.json();
      })
      .then((payload) => setEffective(payload))
      .catch((err) => {
        if (err?.name !== "AbortError") {
          setEffectiveError(err?.message ?? "Failed to load effective config");
        }
      })
      .finally(() => setEffectiveLoading(false));
    return () => controller.abort();
  }, [selectedEpic]);

  useEffect(() => {
    if (!effective) {
      setDraftOverrides({});
      setInitialOverrides({});
      setBaseParamOverrides({});
      return;
    }

    const override = effective.override ?? {};
    const paramOverrides = (override.parameter_overrides as Record<string, unknown>) ?? {};
    const columnSet = new Set(overrideColumns.filter((key) => key !== "parameter_overrides"));
    const columnOverrides: Record<string, unknown> = {};

    Object.keys(override).forEach((key) => {
      if (META_FIELDS.has(key) || key === "parameter_overrides") return;
      if (!columnSet.has(key)) return;
      const value = override[key];
      if (value !== null && value !== undefined) {
        columnOverrides[key] = value;
      }
    });

    const merged = { ...paramOverrides, ...columnOverrides };
    setInitialOverrides(merged);
    setDraftOverrides(merged);
    setBaseParamOverrides(paramOverrides);
  }, [effective, overrideColumns]);

  const overrideCounts = useMemo(() => {
    const counts = new Map<string, number>();
    overrides.forEach((override) => {
      let count = 0;
      if (override.parameter_overrides && typeof override.parameter_overrides === "object") {
        count += Object.keys(override.parameter_overrides as Record<string, unknown>).length;
      }
      Object.entries(override).forEach(([key, value]) => {
        if (META_FIELDS.has(key) || key === "parameter_overrides") return;
        if (value !== null && value !== undefined) {
          count += 1;
        }
      });
      counts.set(override.epic, count);
    });
    return counts;
  }, [overrides]);

  const filteredPairs = useMemo(() => {
    const normalized = pairQuery.trim().toLowerCase();
    return allPairs.filter((pair) => {
      if (showPairOverridesOnly && !overrideCounts.get(pair)) return false;
      return normalized ? pair.toLowerCase().includes(normalized) : true;
    });
  }, [allPairs, overrideCounts, pairQuery, showPairOverridesOnly]);

  const toggleSelected = (epic: string) => {
    setSelected((prev) =>
      prev.includes(epic) ? prev.filter((item) => item !== epic) : [...prev, epic]
    );
  };

  const handleBulk = async (action: string) => {
    if (!selected.length) return;
    if (!updatedBy || !changeReason) {
      alert("Updated by and change reason are required.");
      return;
    }
    await bulkAction(action, selected, {
      updatedBy,
      changeReason
    });
    setSelected([]);
  };

  const handleDelete = async (epic: string) => {
    await deleteOverride(epic, { updatedBy, changeReason });
    await reload();
    if (selectedEpic === epic) {
      setEffective(null);
      setDraftOverrides({});
      setInitialOverrides({});
      setBaseParamOverrides({});
    }
  };

  const activePairOverride = overrides.find((item) => item.epic === selectedEpic);
  const fieldData = effective?.global ?? globalConfig;
  const fieldKeys = fieldData
    ? Object.keys(fieldData).filter((key) => !SYSTEM_FIELDS.has(key))
    : [];

  const searchIndex = useMemo(() => {
    const index: Record<string, string> = {};
    (metadata ?? []).forEach((item) => {
      index[item.parameter_name] = [
        item.parameter_name,
        item.display_name,
        item.description,
        item.help_text,
        item.category,
        item.subcategory
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
    });
    return index;
  }, [metadata]);

  const filteredFields = useSettingsSearch(
    fieldKeys,
    fieldData ?? {},
    {},
    fieldQuery,
    false,
    searchIndex
  );

  const visibleFields = useMemo(() => {
    if (!showFieldOverridesOnly) return filteredFields;
    return filteredFields.filter((field) => hasKey(draftOverrides, field));
  }, [draftOverrides, filteredFields, showFieldOverridesOnly]);

  const sections = useMemo(() => {
    if (!fieldData) return [];
    const filteredSet = new Set(visibleFields);
    const metadataList = metadata ?? [];
    const metadataMap = new Map<string, SmcParameterMetadata>();
    metadataList.forEach((item) => metadataMap.set(item.parameter_name, item));

    const preferredOrder = [
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
      "Other"
    ];

    const sectionsMap = new Map<
      string,
      {
        category: string;
        items: SmcParameterMetadata[];
        subgroups: Map<string, SmcParameterMetadata[]>;
      }
    >();

    const addItem = (item: SmcParameterMetadata) => {
      const category = item.category || "Other";
      const section = sectionsMap.get(category) ?? {
        category,
        items: [],
        subgroups: new Map<string, SmcParameterMetadata[]>()
      };
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

    metadataList.forEach((item) => {
      if (!filteredSet.has(item.parameter_name)) return;
      if (!showAdvanced && item.is_advanced) return;
      addItem(item);
    });

    filteredSet.forEach((field) => {
      if (metadataMap.has(field)) return;
      addItem({
        id: 0,
        parameter_name: field,
        display_name: toLabel(field),
        category: "Other",
        data_type: typeof fieldData[field],
        display_order: 0
      });
    });

    const sortItems = (items: SmcParameterMetadata[]) =>
      [...items].sort((a, b) => {
        const orderA = a.display_order ?? 0;
        const orderB = b.display_order ?? 0;
        if (orderA !== orderB) return orderA - orderB;
        return a.display_name.localeCompare(b.display_name);
      });

    return [...sectionsMap.values()]
      .sort((a, b) => {
        const indexA = preferredOrder.indexOf(a.category);
        const indexB = preferredOrder.indexOf(b.category);
        if (indexA !== -1 || indexB !== -1) {
          return (indexA === -1 ? 999 : indexA) - (indexB === -1 ? 999 : indexB);
        }
        return a.category.localeCompare(b.category);
      })
      .map((section) => ({
        ...section,
        items: sortItems(section.items),
        subgroups: new Map(
          [...section.subgroups.entries()].map(([key, items]) => [
            key,
            sortItems(items)
          ])
        )
      }));
  }, [fieldData, metadata, showAdvanced, visibleFields]);

  const dirtyKeys = useMemo(() => {
    const keys = new Set<string>();
    const allKeys = new Set([
      ...Object.keys(initialOverrides),
      ...Object.keys(draftOverrides)
    ]);
    allKeys.forEach((key) => {
      const hasInitial = hasKey(initialOverrides, key);
      const hasDraft = hasKey(draftOverrides, key);
      if (hasInitial !== hasDraft) {
        keys.add(key);
        return;
      }
      if (!valuesEqual(initialOverrides[key], draftOverrides[key])) {
        keys.add(key);
      }
    });
    return keys;
  }, [draftOverrides, initialOverrides]);

  const toggleOverride = (field: string, next: boolean) => {
    setDraftOverrides((prev) => {
      const copy = { ...prev };
      if (next) {
        copy[field] = fieldData?.[field] ?? "";
      } else {
        delete copy[field];
      }
      return copy;
    });
  };

  const updateOverrideValue = (field: string, value: unknown) => {
    setDraftOverrides((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    if (!selectedEpic) return;
    if (!updatedBy || !changeReason) {
      alert("Updated by and change reason are required.");
      return;
    }

    if (!dirtyKeys.size) {
      alert("No override changes to save.");
      return;
    }

    const columnSet = new Set(overrideColumns.filter((key) => key !== "parameter_overrides"));
    const nextParamOverrides = { ...baseParamOverrides };
    const updates: Record<string, unknown> = {};
    let paramOverridesChanged = false;

    dirtyKeys.forEach((key) => {
      const hasDraft = hasKey(draftOverrides, key);
      const nextValue = hasDraft ? draftOverrides[key] : null;
      if (columnSet.has(key)) {
        updates[key] = nextValue;
      } else {
        if (hasDraft) {
          nextParamOverrides[key] = nextValue;
        } else {
          delete nextParamOverrides[key];
        }
        paramOverridesChanged = true;
      }
    });

    if (paramOverridesChanged) {
      updates.parameter_overrides = nextParamOverrides;
    }

    if (effective?.override && effective.override.updated_at) {
      await saveOverride(
        selectedEpic,
        updates,
        {
          updatedBy,
          changeReason,
          updatedAt: String(effective.override.updated_at)
        }
      );
    } else {
      await createOverride(selectedEpic, updates, {
        updatedBy,
        changeReason
      });
    }

    await reload();
    const refreshed = await fetch(
      apiUrl(`/api/settings/strategy/smc/effective/${selectedEpic}`)
    ).then((res) => res.json());
    setEffective(refreshed);
  };

  if (loading) {
    return <div className="settings-panel">Loading overrides...</div>;
  }
  if (error) {
    return <div className="settings-panel">Error: {error}</div>;
  }

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>SMC Pair Overrides</h1>
        <p>Override global settings on a per-pair basis with live effective previews.</p>
      </div>
      <div className="pair-overrides-layout">
        <div className="pair-overrides-sidebar">
          <div className="pair-overrides-header">
            <SettingsSearch value={pairQuery} onChange={setPairQuery} />
            <div className="settings-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={showPairOverridesOnly}
                  onChange={(event) => setShowPairOverridesOnly(event.target.checked)}
                />
                Show overridden pairs only
              </label>
            </div>
          </div>
          <BulkActionBar
            selectedCount={selected.length}
            onCopyGlobal={() => handleBulk("copy-global")}
            onCopyPair={async () => {
              const sourceEpic = prompt("Source epic to copy from:");
              if (!sourceEpic) return;
              if (!updatedBy || !changeReason) {
                alert("Updated by and change reason are required.");
                return;
              }
              await bulkAction("copy-pair", selected, {
                updatedBy,
                changeReason,
                sourceEpic
              });
              setSelected([]);
            }}
            onReset={() => handleBulk("reset")}
          />
          <div className="pair-list">
            {filteredPairs.map((pair) => {
              const count = overrideCounts.get(pair) ?? 0;
              const isActive = pair === selectedEpic;
              return (
                <button
                  key={pair}
                  className={`pair-card ${isActive ? "active" : ""}`}
                  onClick={() => setSelectedEpic(pair)}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(pair)}
                    onChange={(event) => {
                      event.stopPropagation();
                      toggleSelected(pair);
                    }}
                  />
                  <div className="pair-card-body">
                    <strong>{pair}</strong>
                    <span>{count ? `${count} overrides` : "No overrides"}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="pair-overrides-detail">
          <div className="pair-detail-header">
            <div>
              <h2>{selectedEpic || "Select a pair"}</h2>
              <p>Override the global strategy config for this instrument.</p>
            </div>
            <div className="pair-detail-actions">
              <input
                placeholder="Updated by"
                value={updatedBy}
                onChange={(event) => setUpdatedBy(event.target.value)}
              />
              <input
                placeholder="Change reason"
                value={changeReason}
                onChange={(event) => setChangeReason(event.target.value)}
              />
              {activePairOverride ? (
                <button
                  className="danger"
                  onClick={() => handleDelete(activePairOverride.epic)}
                >
                  Delete Override
                </button>
              ) : null}
              <button className="primary" onClick={handleSave} disabled={!dirtyKeys.size}>
                Save Changes ({dirtyKeys.size})
              </button>
            </div>
          </div>

          {effectiveError ? (
            <div className="settings-placeholder">Error: {effectiveError}</div>
          ) : null}
          {effectiveLoading || metadataLoading ? (
            <div className="settings-placeholder">Loading pair configuration...</div>
          ) : null}

          {effective ? (
            <>
              <div className="pair-detail-meta">
                <div>
                  <strong>Last Updated</strong>
                  <span>
                    {activePairOverride?.updated_at
                      ? new Date(activePairOverride.updated_at).toLocaleString()
                      : "Not overridden"}
                  </span>
                </div>
                <div>
                  <strong>Overrides</strong>
                  <span>{overrideCounts.get(selectedEpic) ?? 0}</span>
                </div>
                <div>
                  <strong>Effective Min Confidence</strong>
                  <span>
                    {String(
                      effective.effective?.min_confidence_threshold ??
                        effective.effective?.min_confidence ??
                        "-"
                    )}
                  </span>
                </div>
              </div>

              <SettingsSearch value={fieldQuery} onChange={setFieldQuery} />
              <div className="settings-toggle">
                <label>
                  <input
                    type="checkbox"
                    checked={showFieldOverridesOnly}
                    onChange={(event) => setShowFieldOverridesOnly(event.target.checked)}
                  />
                  Show overridden fields only
                </label>
              </div>
              <div className="settings-toggle">
                <label>
                  <input
                    type="checkbox"
                    checked={showAdvanced}
                    onChange={(event) => setShowAdvanced(event.target.checked)}
                  />
                  Show advanced parameters
                </label>
              </div>

              {sections.map((section, index) => {
                const fields = [
                  ...section.items.map((item) => item.parameter_name),
                  ...[...section.subgroups.values()].flat().map((item) => item.parameter_name)
                ];
                const changedCount = fields.filter((field) =>
                  hasKey(draftOverrides, field)
                ).length;
                const open = index === 0 || !!fieldQuery.trim() || changedCount > 0;

                return (
                  <details className="settings-section" key={section.category} open={open}>
                    <summary className="settings-section-summary">
                      <div>
                        <h2>{section.category}</h2>
                        <p>{fields.length} fields</p>
                      </div>
                      <div className="settings-section-meta">
                        {changedCount > 0 ? (
                          <span className="settings-section-count">
                            {changedCount} overrides
                          </span>
                        ) : null}
                        {section.subgroups.size > 0 ? (
                          <span className="settings-section-count">
                            {section.subgroups.size} groups
                          </span>
                        ) : null}
                      </div>
                    </summary>
                    <div className="settings-section-body">
                      {section.subgroups.size > 0
                        ? [...section.subgroups.entries()].map(([title, items]) => (
                            <div className="settings-subgroup" key={title}>
                              <div className="settings-subgroup-title">{title}</div>
                              <div className="override-grid">
                                {items.map((item) => {
                                  const descriptionParts = [
                                    item.help_text ?? item.description
                                  ];
                                  if (item.requires_restart) {
                                    descriptionParts.push("Requires restart");
                                  }
                                  return (
                                    <OverrideField
                                      key={item.parameter_name}
                                      name={item.parameter_name}
                                      label={item.display_name || toLabel(item.parameter_name)}
                                      description={descriptionParts
                                        .filter(Boolean)
                                        .join(" · ")}
                                      unit={item.unit ?? undefined}
                                      dataType={item.data_type}
                                      globalValue={effective.global[item.parameter_name]}
                                      effectiveValue={effective.effective[item.parameter_name]}
                                      overrideValue={draftOverrides[item.parameter_name]}
                                      isOverridden={hasKey(
                                        draftOverrides,
                                        item.parameter_name
                                      )}
                                      onToggle={(next) =>
                                        toggleOverride(item.parameter_name, next)
                                      }
                                      onChange={(value) =>
                                        updateOverrideValue(item.parameter_name, value)
                                      }
                                    />
                                  );
                                })}
                              </div>
                            </div>
                          ))
                        : section.items.map((item) => {
                            const descriptionParts = [
                              item.help_text ?? item.description
                            ];
                            if (item.requires_restart) {
                              descriptionParts.push("Requires restart");
                            }
                            return (
                              <OverrideField
                                key={item.parameter_name}
                                name={item.parameter_name}
                                label={item.display_name || toLabel(item.parameter_name)}
                                description={descriptionParts.filter(Boolean).join(" · ")}
                                unit={item.unit ?? undefined}
                                dataType={item.data_type}
                                globalValue={effective.global[item.parameter_name]}
                                effectiveValue={effective.effective[item.parameter_name]}
                                overrideValue={draftOverrides[item.parameter_name]}
                                isOverridden={hasKey(draftOverrides, item.parameter_name)}
                                onToggle={(next) =>
                                  toggleOverride(item.parameter_name, next)
                                }
                                onChange={(value) =>
                                  updateOverrideValue(item.parameter_name, value)
                                }
                              />
                            );
                          })}
                    </div>
                  </details>
                );
              })}
            </>
          ) : (
            <div className="settings-placeholder">Select a pair to begin.</div>
          )}
        </div>
      </div>
    </div>
  );
}
