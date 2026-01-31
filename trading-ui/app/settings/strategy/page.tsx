"use client";

import { useEffect, useMemo, useState } from "react";
import SettingsField from "../../../components/settings/SettingsField";
import SettingsForm from "../../../components/settings/SettingsForm";
import SettingsGroup from "../../../components/settings/SettingsGroup";
import SettingsSearch from "../../../components/settings/SettingsSearch";
import ConflictModal from "../../../components/settings/ConflictModal";
import { useSmcConfig } from "../../../hooks/settings/useSmcConfig";
import { useSettingsSearch } from "../../../hooks/settings/useSettingsSearch";
import { logTelemetry } from "../../../lib/settings/telemetry";

function toLabel(value: string) {
  return value.replace(/_/g, " ");
}

export default function SmcGlobalSettingsPage() {
  const [query, setQuery] = useState("");
  const [modifiedOnly, setModifiedOnly] = useState(false);
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

  const data = effectiveData as Record<string, unknown> | null;
  const keys = data ? Object.keys(data).filter((key) => !["id", "created_at"].includes(key)) : [];
  const filtered = useSettingsSearch(keys, data, {}, query, modifiedOnly);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy" });
  }, []);

  const content = useMemo(() => {
    if (!data) return null;
    return filtered.map((field) => (
      <SettingsField
        key={field}
        name={field}
        label={toLabel(field)}
        value={data[field]}
        onChange={(value) => updateField(field, value)}
      />
    ));
  }, [data, filtered, updateField]);

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
