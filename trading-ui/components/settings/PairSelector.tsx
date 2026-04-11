"use client";

import { useState, useRef, useEffect } from "react";
import { epicToDisplayName } from "../../lib/settings/epicDisplay";

interface PairSelectorProps {
  pairs: string[];
  value: string;
  overrideCounts: Map<string, number>;
  onChange: (epic: string) => void;
}

export default function PairSelector({
  pairs,
  value,
  overrideCounts,
  onChange,
}: PairSelectorProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filtered = pairs.filter((epic) => {
    if (!query.trim()) return true;
    const display = epicToDisplayName(epic).toLowerCase();
    return (
      display.includes(query.toLowerCase()) ||
      epic.toLowerCase().includes(query.toLowerCase())
    );
  });

  const displayValue = value ? epicToDisplayName(value) : "Select pair…";
  const activeCount = value ? (overrideCounts.get(value) ?? 0) : 0;

  return (
    <div className="pair-selector" ref={ref}>
      <button
        type="button"
        className={`pair-selector-trigger ${open ? "open" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="pair-selector-value">{displayValue}</span>
        {activeCount > 0 ? (
          <span className="pair-selector-badge">{activeCount}</span>
        ) : null}
        <span className="pair-selector-arrow">▾</span>
      </button>

      {open ? (
        <div className="pair-selector-dropdown">
          <div className="pair-selector-search">
            <input
              type="text"
              placeholder="Search pairs…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
          </div>
          <div className="pair-selector-list">
            {filtered.map((epic) => {
              const count = overrideCounts.get(epic) ?? 0;
              const isActive = epic === value;
              return (
                <button
                  key={epic}
                  type="button"
                  className={`pair-selector-option ${isActive ? "active" : ""}`}
                  onClick={() => {
                    onChange(epic);
                    setOpen(false);
                    setQuery("");
                  }}
                >
                  <span className="pair-selector-option-name">
                    {epicToDisplayName(epic)}
                  </span>
                  <span className="pair-selector-option-epic">{epic}</span>
                  {count > 0 ? (
                    <span className="pair-selector-option-badge">{count} overrides</span>
                  ) : null}
                </button>
              );
            })}
            {filtered.length === 0 ? (
              <div className="pair-selector-empty">No pairs match "{query}"</div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
