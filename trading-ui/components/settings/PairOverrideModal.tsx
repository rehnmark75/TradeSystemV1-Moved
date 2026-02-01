"use client";

import { useEffect, useMemo, useState } from "react";

interface PairOverrideModalProps {
  open: boolean;
  epic?: string;
  initialOverrides?: Record<string, unknown>;
  onClose: () => void;
  onSave: (payload: { epic: string; overrides: Record<string, unknown> }) => void;
}

export default function PairOverrideModal({
  open,
  epic,
  initialOverrides,
  onClose,
  onSave
}: PairOverrideModalProps) {
  const [localEpic, setLocalEpic] = useState(epic ?? "");
  const [showJson, setShowJson] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [rows, setRows] = useState<
    Array<{ key: string; type: "number" | "boolean" | "string"; value: string }>
  >([]);

  useEffect(() => {
    setLocalEpic(epic ?? "");
    const overrides = initialOverrides ?? {};
    setJsonText(JSON.stringify(overrides, null, 2));
    const nextRows = Object.entries(overrides).map(([key, value]) => {
      const valueType: "number" | "boolean" | "string" =
        typeof value === "number"
          ? "number"
          : typeof value === "boolean"
          ? "boolean"
          : "string";
      return {
        key,
        type: valueType,
        value: value === null || value === undefined ? "" : String(value)
      };
    });
    setRows(nextRows);
  }, [epic, initialOverrides, open]);

  if (!open) return null;

  const parsedOverrides = useMemo(() => {
    if (showJson) {
      try {
        return jsonText.trim() ? JSON.parse(jsonText) : {};
      } catch {
        return null;
      }
    }
    const next: Record<string, unknown> = {};
    rows.forEach((row) => {
      if (!row.key) return;
      if (row.type === "number") {
        const parsed = Number(row.value);
        next[row.key] = Number.isNaN(parsed) ? row.value : parsed;
        return;
      }
      if (row.type === "boolean") {
        next[row.key] = row.value === "true";
        return;
      }
      next[row.key] = row.value;
    });
    return next;
  }, [rows, jsonText, showJson]);

  const handleSave = () => {
    if (!localEpic.trim()) {
      alert("Epic is required.");
      return;
    }
    if (parsedOverrides === null) {
      alert("Invalid JSON in overrides.");
      return;
    }
    onSave({ epic: localEpic.trim(), overrides: parsedOverrides });
  };

  const updateRow = (index: number, field: "key" | "type" | "value", value: string) => {
    setRows((prev) =>
      prev.map((row, idx) =>
        idx === index ? { ...row, [field]: value } : row
      )
    );
  };

  const addRow = () => {
    setRows((prev) => [...prev, { key: "", type: "number", value: "" }]);
  };

  const removeRow = (index: number) => {
    setRows((prev) => prev.filter((_, idx) => idx !== index));
  };

  return (
    <div className="conflict-modal">
      <div className="conflict-modal-content">
        <h3>{epic ? `Edit ${epic}` : "Create Override"}</h3>
        <input
          placeholder="Epic (e.g. CS.D.EURUSD.MINI.IP)"
          value={localEpic}
          onChange={(event) => setLocalEpic(event.target.value)}
        />
        <div className="settings-toggle">
          <label>
            <input
              type="checkbox"
              checked={showJson}
              onChange={(event) => setShowJson(event.target.checked)}
            />
            Advanced JSON editor
          </label>
        </div>
        {showJson ? (
          <textarea
            value={jsonText}
            rows={10}
            onChange={(event) => setJsonText(event.target.value)}
          />
        ) : (
          <div className="override-editor">
            <div className="override-row override-header">
              <span>Field</span>
              <span>Type</span>
              <span>Value</span>
              <span />
            </div>
            {rows.map((row, index) => (
              <div key={`${row.key}-${index}`} className="override-row">
                <input
                  placeholder="field_name"
                  value={row.key}
                  onChange={(event) => updateRow(index, "key", event.target.value)}
                />
                <select
                  value={row.type}
                  onChange={(event) =>
                    updateRow(index, "type", event.target.value as "number" | "boolean" | "string")
                  }
                >
                  <option value="number">Number</option>
                  <option value="boolean">Boolean</option>
                  <option value="string">String</option>
                </select>
                {row.type === "boolean" ? (
                  <select
                    value={row.value || "false"}
                    onChange={(event) => updateRow(index, "value", event.target.value)}
                  >
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <input
                    value={row.value}
                    onChange={(event) => updateRow(index, "value", event.target.value)}
                  />
                )}
                <button onClick={() => removeRow(index)}>Remove</button>
              </div>
            ))}
            <button className="primary" onClick={addRow}>
              Add Field
            </button>
          </div>
        )}
        <div className="conflict-modal-actions">
          <button className="primary" onClick={handleSave}>
            Save Override
          </button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}
