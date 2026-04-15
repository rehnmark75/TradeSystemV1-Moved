"use client";

import { useEffect, useState } from "react";
import { useSnapshots } from "../../hooks/settings/useSnapshots";
import SnapshotDiffView from "./SnapshotDiffView";
import type { ConfigSnapshot, SnapshotDiff } from "../../types/settings";

interface SnapshotPanelProps {
  configSet: string;
  onClose: () => void;
  onRestored: () => void;
}

interface DiffState {
  snapshotId: number;
  snapshotName: string;
  snapshotDate: string;
  diff: SnapshotDiff[];
  changedCount: number;
}

export default function SnapshotPanel({ configSet, onClose, onRestored }: SnapshotPanelProps) {
  const {
    snapshots,
    loading,
    error,
    load,
    createSnapshot,
    deleteSnapshot,
    restoreSnapshot,
    compareSnapshot,
  } = useSnapshots(configSet);

  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [diffState, setDiffState] = useState<DiffState | null>(null);
  const [actionLoading, setActionLoading] = useState<number | null>(null);

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      await createSnapshot(newName.trim(), newDesc.trim() || undefined);
      setNewName("");
      setNewDesc("");
      setShowCreate(false);
    } catch (err: any) {
      setCreateError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this snapshot? This cannot be undone.")) return;
    setActionLoading(id);
    try {
      await deleteSnapshot(id);
    } finally {
      setActionLoading(null);
    }
  };

  const handleCompare = async (snapshot: ConfigSnapshot) => {
    setActionLoading(snapshot.id);
    try {
      const result = await compareSnapshot(snapshot.id);
      setDiffState({
        snapshotId: snapshot.id,
        snapshotName: snapshot.snapshot_name,
        snapshotDate: snapshot.created_at,
        diff: result.diff,
        changedCount: result.changed_count,
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleRestore = async (id: number, name: string) => {
    if (!confirm(`Restore snapshot "${name}"? This will overwrite the current ${configSet} config.`)) return;
    setActionLoading(id);
    try {
      await restoreSnapshot(id);
      setDiffState(null);
      onRestored();
      onClose();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setActionLoading(null);
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  };

  return (
    <>
      <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
        <div className="snapshot-panel">
          <div className="snapshot-panel-header">
            <h2>Config Snapshots</h2>
            <button type="button" className="modal-close" onClick={onClose}>×</button>
          </div>
          <p className="snapshot-panel-desc">
            Snapshots capture the full current strategy config so you can compare and restore at any time.
          </p>

          <div className="snapshot-panel-actions">
            <button
              type="button"
              className="btn-primary"
              onClick={() => setShowCreate((v) => !v)}
            >
              + New snapshot
            </button>
          </div>

          {showCreate ? (
            <div className="snapshot-create-form">
              <input
                type="text"
                placeholder="Snapshot name (e.g. apr-11-tight-sl)"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
              />
              <input
                type="text"
                placeholder="Description (optional)"
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
              />
              {createError ? (
                <div className="snapshot-create-error">{createError}</div>
              ) : null}
              <div className="snapshot-create-btns">
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => {
                    setShowCreate(false);
                    setCreateError(null);
                    setNewName("");
                    setNewDesc("");
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className={`btn-primary ${!newName.trim() || creating ? "disabled" : ""}`}
                  disabled={!newName.trim() || creating}
                  onClick={handleCreate}
                >
                  {creating ? "Saving…" : "Save snapshot"}
                </button>
              </div>
            </div>
          ) : null}

          {loading ? (
            <div className="snapshot-loading">Loading snapshots…</div>
          ) : error ? (
            <div className="snapshot-error">{error}</div>
          ) : snapshots.length === 0 ? (
            <div className="snapshot-empty">
              No snapshots yet. Create one to capture the current config.
            </div>
          ) : (
            <div className="snapshot-list">
              {snapshots.map((snap) => (
                <div key={snap.id} className="snapshot-item">
                  <div className="snapshot-item-info">
                    <div className="snapshot-item-name">{snap.snapshot_name}</div>
                    {snap.description ? (
                      <div className="snapshot-item-desc">{snap.description}</div>
                    ) : null}
                    <div className="snapshot-item-meta">
                      {formatDate(snap.created_at)}
                      {snap.created_by ? ` · by ${snap.created_by}` : ""}
                    </div>
                  </div>
                  <div className="snapshot-item-actions">
                    <button
                      type="button"
                      className="btn-ghost btn-sm"
                      onClick={() => handleCompare(snap)}
                      disabled={actionLoading === snap.id}
                    >
                      Compare
                    </button>
                    <button
                      type="button"
                      className="btn-secondary btn-sm"
                      onClick={() => handleRestore(snap.id, snap.snapshot_name)}
                      disabled={actionLoading === snap.id}
                    >
                      Restore
                    </button>
                    <button
                      type="button"
                      className="btn-danger btn-sm"
                      onClick={() => handleDelete(snap.id)}
                      disabled={actionLoading === snap.id}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {diffState ? (
        <SnapshotDiffView
          snapshotName={diffState.snapshotName}
          snapshotDate={diffState.snapshotDate}
          diff={diffState.diff}
          changedCount={diffState.changedCount}
          onRestore={() => handleRestore(diffState.snapshotId, diffState.snapshotName)}
          onClose={() => setDiffState(null)}
        />
      ) : null}
    </>
  );
}
