"use client";

import { useEffect, useMemo, useState } from "react";
import SettingsField from "../../../../components/settings/SettingsField";
import SettingsForm from "../../../../components/settings/SettingsForm";
import SettingsGroup from "../../../../components/settings/SettingsGroup";
import SettingsSearch from "../../../../components/settings/SettingsSearch";
import ConflictModal from "../../../../components/settings/ConflictModal";
import { scannerCategories } from "../../../../lib/settings/scannerCategories";
import { useScannerConfig } from "../../../../hooks/settings/useScannerConfig";
import { useSettingsSearch } from "../../../../hooks/settings/useSettingsSearch";
import { logTelemetry } from "../../../../lib/settings/telemetry";

interface ScannerCategoryPageProps {
  params: { category: string };
}

function toLabel(value: string) {
  return value.replace(/_/g, " ");
}

export default function ScannerCategoryPage({ params }: ScannerCategoryPageProps) {
  const [query, setQuery] = useState("");
  const [modifiedOnly, setModifiedOnly] = useState(false);
  const [updatedBy, setUpdatedBy] = useState("");
  const [changeReason, setChangeReason] = useState("");
  const {
    effectiveData,
    defaults,
    loading,
    error,
    changes,
    updateField,
    saveChanges,
    resetChanges,
    conflict,
    setConflict,
    setData,
    setChanges
  } = useScannerConfig();

  const categoryKey = params.category;
  const fields = scannerCategories[categoryKey] ?? [];
  const dataKeys = effectiveData ? Object.keys(effectiveData as Record<string, unknown>) : [];
  const knownFields = Object.values(scannerCategories).flat();
  const unassignedFields = dataKeys.filter(
    (key) => !knownFields.includes(key) && !["id", "version", "created_at"].includes(key)
  );
  const finalFields =
    categoryKey === "core" ? [...fields, ...unassignedFields] : fields;
  const data = effectiveData as Record<string, unknown> | null;
  const filtered = useSettingsSearch(finalFields, data, defaults, query, modifiedOnly);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: `/settings/scanner/${categoryKey}` });
  }, [categoryKey]);

  const content = useMemo(() => {
    if (!data) return null;
    return filtered.map((field) => (
      <SettingsField
        key={field}
        name={field}
        label={toLabel(field)}
        value={data[field]}
        defaultValue={defaults[field]}
        override={defaults[field] !== undefined && data[field] !== defaults[field]}
        pending={field in changes}
        onChange={(value) => updateField(field, value)}
      />
    ));
  }, [data, filtered, defaults, changes, updateField]);

  if (loading) {
    return <div className="settings-panel">Loading settings...</div>;
  }

  if (categoryKey === "audit") {
    return (
      <div className="settings-panel">
        <div className="settings-hero">
          <h1>Scanner Audit</h1>
          <p>View the full audit trail in the audit section.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>Scanner Settings</h1>
        <p>Category: {categoryKey}</p>
      </div>
      {error ? <div className="settings-placeholder">Error: {error}</div> : null}
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
      <SettingsGroup title="Fields">{content}</SettingsGroup>
      <SettingsForm
        title="Scanner Updates"
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
            setData(conflict as any);
            resetChanges();
            setConflict(null);
            return;
          }
          if (mergedChanges) {
            if (!updatedBy.trim() || !changeReason.trim()) {
              alert("Updated by and change reason are required.");
              return;
            }
            setData(conflict as any);
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
