"use client";

import { useEffect, useMemo, useState } from "react";
import BulkActionBar from "../../../../components/settings/BulkActionBar";
import SettingsSearch from "../../../../components/settings/SettingsSearch";
import PairOverrideModal from "../../../../components/settings/PairOverrideModal";
import { usePairOverrides } from "../../../../hooks/settings/usePairOverrides";
import { logTelemetry } from "../../../../lib/settings/telemetry";

export default function SmcPairOverridesPage() {
  const { overrides, loading, error, bulkAction, deleteOverride, saveOverride, createOverride } =
    usePairOverrides();
  const [selected, setSelected] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [updatedBy, setUpdatedBy] = useState("");
  const [changeReason, setChangeReason] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<{
    epic?: string;
    overrides?: Record<string, unknown>;
    updatedAt?: string;
  } | null>(null);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy/pairs" });
  }, []);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return overrides.filter((item) =>
      normalized ? item.epic.toLowerCase().includes(normalized) : true
    );
  }, [overrides, query]);

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
  };

  const openCreate = () => {
    setEditing(null);
    setModalOpen(true);
  };

  const openEdit = (epic: string) => {
    const current = overrides.find((item) => item.epic === epic);
    setEditing({
      epic,
      overrides: (current?.parameter_overrides as Record<string, unknown>) ?? {},
      updatedAt: current?.updated_at
    });
    setModalOpen(true);
  };

  const handleSave = async (payload: { epic: string; overrides: Record<string, unknown> }) => {
    if (!payload.epic) return;
    if (!updatedBy || !changeReason) {
      alert("Updated by and change reason are required.");
      return;
    }
    if (editing?.epic && editing.updatedAt) {
      await saveOverride(
        payload.epic,
        { parameter_overrides: payload.overrides },
        {
          updatedBy,
          changeReason,
          updatedAt: editing.updatedAt
        }
      );
    } else {
      await createOverride(payload.epic, payload.overrides, {
        updatedBy,
        changeReason
      });
    }
    setModalOpen(false);
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
        <p>Manage pair-specific overrides and bulk actions.</p>
      </div>
      <SettingsSearch value={query} onChange={setQuery} />
      <div className="settings-form-actions">
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
        <button className="primary" onClick={openCreate}>
          Create Override
        </button>
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
      <div className="overrides-grid">
        <div className="overrides-header">
          <span />
          <span>Epic</span>
          <span>Updated At</span>
          <span>Actions</span>
        </div>
        {filtered.map((override) => (
          <div key={override.epic} className="overrides-row">
            <input
              type="checkbox"
              checked={selected.includes(override.epic)}
              onChange={() => toggleSelected(override.epic)}
            />
            <span>{override.epic}</span>
            <span>{new Date(override.updated_at).toLocaleString()}</span>
            <div className="overrides-actions">
              <button onClick={() => openEdit(override.epic)}>Edit</button>
              <button onClick={() => handleDelete(override.epic)}>Delete</button>
            </div>
          </div>
        ))}
      </div>
      <PairOverrideModal
        open={modalOpen}
        epic={editing?.epic}
        initialOverrides={editing?.overrides}
        onClose={() => setModalOpen(false)}
        onSave={handleSave}
      />
    </div>
  );
}
