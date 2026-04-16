"use client";

import { useEffect, useRef, useState } from "react";
import {
  ALL_STAGES,
  CATEGORIES,
  CATEGORY_ORDER,
  describeStage,
  type CategoryKey,
} from "../lib/rejectionStyles";

interface Props {
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  counts?: Record<string, number>;
}

export default function RejectionStageFilter({ selected, onChange, counts }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  const totalSelected = selected.size;
  const totalStages = ALL_STAGES.length;

  function selectAll() {
    onChange(new Set(ALL_STAGES));
  }
  function selectNone() {
    onChange(new Set());
  }

  function toggleStage(stage: string) {
    const next = new Set(selected);
    if (next.has(stage)) next.delete(stage);
    else next.add(stage);
    onChange(next);
  }

  function toggleCategory(key: CategoryKey) {
    const stages = CATEGORIES[key].stages;
    const allOn = stages.every((s) => selected.has(s));
    const next = new Set(selected);
    if (allOn) {
      for (const s of stages) next.delete(s);
    } else {
      for (const s of stages) next.add(s);
    }
    onChange(next);
  }

  return (
    <div ref={rootRef} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          padding: "4px 10px",
          borderRadius: "4px",
          border: "1px solid #334155",
          background: "#1e293b",
          color: "#e2e8f0",
          cursor: "pointer",
          fontSize: "12px",
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}
      >
        <span>
          Rejections:{" "}
          <span style={{ color: "#60a5fa" }}>
            {totalSelected} / {totalStages}
          </span>
        </span>
        <span style={{ color: "#64748b", fontSize: "10px" }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            left: 0,
            zIndex: 20,
            minWidth: "380px",
            maxHeight: "500px",
            overflowY: "auto",
            background: "#0f172a",
            border: "1px solid #334155",
            borderRadius: "6px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            padding: "8px",
            fontSize: "12px",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "6px",
              marginBottom: "6px",
              paddingBottom: "6px",
              borderBottom: "1px solid #1e293b",
            }}
          >
            <button onClick={selectAll} style={btnStyle}>
              All
            </button>
            <button onClick={selectNone} style={btnStyle}>
              None
            </button>
            <span style={{ marginLeft: "auto", color: "#64748b", alignSelf: "center" }}>
              {totalSelected} selected
            </span>
          </div>

          {CATEGORY_ORDER.map((key) => {
            const cat = CATEGORIES[key];
            const onCount = cat.stages.filter((s) => selected.has(s)).length;
            const allOn = onCount === cat.stages.length;
            const indeterminate = onCount > 0 && !allOn;
            return (
              <div key={key} style={{ marginBottom: "6px" }}>
                <label
                  title={cat.description}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    cursor: "pointer",
                    padding: "3px 4px",
                    borderRadius: "3px",
                    background: "#111827",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={allOn}
                    ref={(el) => {
                      if (el) el.indeterminate = indeterminate;
                    }}
                    onChange={() => toggleCategory(key)}
                  />
                  <span
                    style={{
                      width: "10px",
                      height: "10px",
                      borderRadius: "50%",
                      background: cat.color,
                      display: "inline-block",
                    }}
                  />
                  <span style={{ color: "#e2e8f0", fontWeight: 600 }}>{cat.label}</span>
                  <span style={{ color: "#64748b", marginLeft: "auto" }}>
                    {onCount}/{cat.stages.length}
                  </span>
                </label>
                <div style={{ paddingLeft: "22px", marginTop: "2px" }}>
                  {cat.stages.map((stage) => {
                    const count = counts?.[stage];
                    const desc = describeStage(stage);
                    return (
                      <label
                        key={stage}
                        title={desc}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "6px",
                          cursor: "pointer",
                          padding: "2px 0",
                          color: "#cbd5e1",
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={selected.has(stage)}
                          onChange={() => toggleStage(stage)}
                        />
                        <span style={{ fontFamily: "monospace", fontSize: "11px" }}>
                          {stage}
                        </span>
                        <span
                          style={{
                            color: "#64748b",
                            fontSize: "10px",
                            flex: 1,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          — {desc}
                        </span>
                        {count != null && (
                          <span style={{ color: "#64748b", fontSize: "11px" }}>
                            {count}
                          </span>
                        )}
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "3px 8px",
  borderRadius: "3px",
  border: "1px solid #334155",
  background: "#1e293b",
  color: "#cbd5e1",
  cursor: "pointer",
  fontSize: "11px",
};
