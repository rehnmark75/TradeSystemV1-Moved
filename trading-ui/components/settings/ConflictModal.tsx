"use client";

import { useEffect, useMemo, useState } from "react";

interface ConflictModalProps {
  open: boolean;
  current?: Record<string, unknown> | null;
  pending?: Record<string, unknown>;
  onClose: () => void;
  onResolve: (resolution: {
    action: "overwrite" | "merge" | "discard";
    mergedChanges?: Record<string, unknown>;
  }) => void;
}

export default function ConflictModal({
  open,
  current,
  pending = {},
  onClose,
  onResolve
}: ConflictModalProps) {
  const keys = useMemo(() => Object.keys(pending), [pending]);
  const [choices, setChoices] = useState<Record<string, "mine" | "server">>({});

  useEffect(() => {
    setChoices({});
  }, [open, keys.join(",")]);

  if (!open) return null;

  const handleChoice = (key: string, value: "mine" | "server") => {
    setChoices((prev) => ({ ...prev, [key]: value }));
  };

  const merged = keys.reduce<Record<string, unknown>>((acc, key) => {
    const choice = choices[key] ?? "mine";
    acc[key] = choice === "server" ? current?.[key] : pending[key];
    return acc;
  }, {});

  return (
    <div className="conflict-modal">
      <div className="conflict-modal-content">
        <h3>Conflict Detected</h3>
        <p>Settings were updated by someone else. Choose how to resolve.</p>
        <div className="conflict-grid">
          <div className="conflict-row conflict-header">
            <span>Field</span>
            <span>Your Value</span>
            <span>Server Value</span>
            <span>Pick</span>
          </div>
          {keys.map((key) => (
            <div key={key} className="conflict-row">
              <span>{key}</span>
              <span>{JSON.stringify(pending[key])}</span>
              <span>{JSON.stringify(current?.[key])}</span>
              <span>
                <label>
                  <input
                    type="radio"
                    name={`choice-${key}`}
                    checked={(choices[key] ?? "mine") === "mine"}
                    onChange={() => handleChoice(key, "mine")}
                  />
                  Mine
                </label>
                <label>
                  <input
                    type="radio"
                    name={`choice-${key}`}
                    checked={choices[key] === "server"}
                    onChange={() => handleChoice(key, "server")}
                  />
                  Server
                </label>
              </span>
            </div>
          ))}
        </div>
        <div className="conflict-modal-actions">
          <button
            className="primary"
            onClick={() => onResolve({ action: "overwrite", mergedChanges: pending })}
          >
            Overwrite
          </button>
          <button
            onClick={() => onResolve({ action: "merge", mergedChanges: merged })}
          >
            Merge & Save
          </button>
          <button onClick={() => onResolve({ action: "discard" })}>
            Discard Mine
          </button>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
