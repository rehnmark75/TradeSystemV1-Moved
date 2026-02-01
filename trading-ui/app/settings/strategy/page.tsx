"use client";

import { useEffect, useMemo, useState } from "react";
import SettingsField from "../../../components/settings/SettingsField";
import SettingsForm from "../../../components/settings/SettingsForm";
import SettingsSearch from "../../../components/settings/SettingsSearch";
import ConflictModal from "../../../components/settings/ConflictModal";
import { useSmcConfig } from "../../../hooks/settings/useSmcConfig";
import { useSmcMetadata } from "../../../hooks/settings/useSmcMetadata";
import { useSettingsSearch } from "../../../hooks/settings/useSettingsSearch";
import { logTelemetry } from "../../../lib/settings/telemetry";
import type { SmcParameterMetadata } from "../../../types/settings";

function toLabel(value: string) {
  return value.replace(/_/g, " ");
}

export default function SmcGlobalSettingsPage() {
  const [query, setQuery] = useState("");
  const [modifiedOnly, setModifiedOnly] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [updatedBy, setUpdatedBy] = useState("");
  const [changeReason, setChangeReason] = useState("");
  const {
    effectiveData,
    loading,
    error,
    changes,
    updateField,
    saveChanges,
    resetChanges,
    conflict,
    setConflict,
    setChanges
  } = useSmcConfig();
  const {
    metadata,
    loading: metadataLoading,
    error: metadataError
  } = useSmcMetadata();

  const data = effectiveData as Record<string, unknown> | null;
  const keys = data ? Object.keys(data).filter((key) => !["id", "created_at"].includes(key)) : [];
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
  const filtered = useSettingsSearch(keys, data, {}, query, modifiedOnly, searchIndex);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy" });
  }, []);

  const sections = useMemo(() => {
    if (!data) return [];
    const filteredSet = new Set(filtered);
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
        data_type: typeof data[field],
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
  }, [data, filtered, metadata, showAdvanced]);

  const changeKeys = useMemo(() => new Set(Object.keys(changes)), [changes]);
  const safeData = data ?? {};

  if (loading) {
    return <div className="settings-panel">Loading settings...</div>;
  }

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>SMC Global Settings</h1>
        <p>Global configuration for SMC Simple strategy.</p>
      </div>
      {error ? <div className="settings-placeholder">Error: {error}</div> : null}
      {metadataError ? (
        <div className="settings-placeholder">Metadata error: {metadataError}</div>
      ) : null}
      <SettingsSearch value={query} onChange={setQuery} />
      <div className="settings-toggle">
        <label>
          <input
            type="checkbox"
            checked={modifiedOnly}
            onChange={(event) => setModifiedOnly(event.target.checked)}
          />
          Show modified only
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
      {(metadataLoading || sections.length === 0) && !error ? (
        <div className="settings-placeholder">Loading SMC metadata...</div>
      ) : null}
      {sections.map((section, index) => {
        const fields = [
          ...section.items.map((item) => item.parameter_name),
          ...[...section.subgroups.values()].flat().map((item) => item.parameter_name)
        ];
        const changedCount = fields.filter((field) => changeKeys.has(field)).length;
        const open =
          index === 0 || !!query.trim() || modifiedOnly || changedCount > 0;

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
                    {changedCount} changed
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
                      <div className="settings-section-grid">
                        {items.map((item) => {
                          const descriptionParts = [item.help_text ?? item.description];
                          if (item.requires_restart) {
                            descriptionParts.push("Requires restart");
                          }
                          return (
                            <SettingsField
                              key={item.parameter_name}
                              name={item.parameter_name}
                              label={item.display_name || toLabel(item.parameter_name)}
                              value={safeData[item.parameter_name]}
                              defaultValue={item.default_value ?? undefined}
                              description={descriptionParts.filter(Boolean).join(" · ")}
                              unit={item.unit ?? undefined}
                              pending={changeKeys.has(item.parameter_name)}
                              onChange={(value) => updateField(item.parameter_name, value)}
                            />
                          );
                        })}
                      </div>
                    </div>
                  ))
                : section.items.map((item) => {
                    const descriptionParts = [item.help_text ?? item.description];
                    if (item.requires_restart) {
                      descriptionParts.push("Requires restart");
                    }
                    return (
                      <SettingsField
                        key={item.parameter_name}
                        name={item.parameter_name}
                        label={item.display_name || toLabel(item.parameter_name)}
                        value={safeData[item.parameter_name]}
                        defaultValue={item.default_value ?? undefined}
                        description={descriptionParts.filter(Boolean).join(" · ")}
                        unit={item.unit ?? undefined}
                        pending={changeKeys.has(item.parameter_name)}
                        onChange={(value) => updateField(item.parameter_name, value)}
                      />
                    );
                  })}
            </div>
          </details>
        );
      })}
      <SettingsForm
        title="SMC Updates"
        changes={changes}
        updatedBy={updatedBy}
        changeReason={changeReason}
        onUpdatedByChange={setUpdatedBy}
        onChangeReasonChange={setChangeReason}
        onSave={({ updatedBy, changeReason }) =>
          saveChanges({ updatedBy, changeReason })
        }
        onRevert={resetChanges}
        onDiscard={resetChanges}
      />
      <ConflictModal
        open={!!conflict}
        current={conflict}
        pending={changes}
        onClose={() => setConflict(null)}
        onResolve={async ({ action, mergedChanges }) => {
          if (!conflict) return;
          if (action === "discard") {
            resetChanges();
            setConflict(null);
            return;
          }
          if (mergedChanges) {
            if (!updatedBy.trim() || !changeReason.trim()) {
              alert("Updated by and change reason are required.");
              return;
            }
            setChanges(mergedChanges);
            await saveChanges(
              { updatedBy, changeReason },
              mergedChanges,
              (conflict as any).updated_at
            );
            setConflict(null);
          }
        }}
      />
    </div>
  );
}
