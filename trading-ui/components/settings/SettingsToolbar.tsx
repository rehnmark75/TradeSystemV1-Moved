"use client";

import SegmentedControl from "./SegmentedControl";
import PairSelector from "./PairSelector";

export type SettingsMode = "global" | "pair";

interface FilterState {
  modifiedOnly: boolean;
  showAdvanced: boolean;
  overriddenOnly: boolean;
  requiresRestartOnly: boolean;
}

interface SettingsToolbarProps {
  mode: SettingsMode;
  onModeChange: (mode: SettingsMode) => void;
  // Pair mode
  pairs?: string[];
  selectedPair?: string;
  pairOverrideCounts?: Map<string, number>;
  onPairChange?: (epic: string) => void;
  // Search
  query: string;
  onQueryChange: (q: string) => void;
  // Filters
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  // Actions
  pendingCount: number;
  onSave: () => void;
  onSnapshot: () => void;
  onDiscard: () => void;
  showSnapshotAction?: boolean;
}

const MODE_OPTIONS = [
  { value: "global" as SettingsMode, label: "Global Config" },
  { value: "pair" as SettingsMode, label: "Pair Override" },
];

export default function SettingsToolbar({
  mode,
  onModeChange,
  pairs = [],
  selectedPair = "",
  pairOverrideCounts = new Map(),
  onPairChange,
  query,
  onQueryChange,
  filters,
  onFiltersChange,
  pendingCount,
  onSave,
  onSnapshot,
  onDiscard,
  showSnapshotAction = true,
}: SettingsToolbarProps) {
  const toggleFilter = (key: keyof FilterState) => {
    onFiltersChange({ ...filters, [key]: !filters[key] });
  };

  return (
    <div className="settings-toolbar">
      <div className="settings-toolbar-left">
        <SegmentedControl
          options={MODE_OPTIONS}
          value={mode}
          onChange={onModeChange}
          size="sm"
        />

        {mode === "pair" && onPairChange ? (
          <PairSelector
            pairs={pairs}
            value={selectedPair}
            overrideCounts={pairOverrideCounts}
            onChange={onPairChange}
          />
        ) : null}

        <div className="settings-toolbar-search">
          <span className="settings-toolbar-search-icon">⌕</span>
          <input
            type="text"
            placeholder="Search parameters…"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
          />
          {query ? (
            <button
              type="button"
              className="settings-toolbar-search-clear"
              onClick={() => onQueryChange("")}
            >
              ×
            </button>
          ) : null}
        </div>
      </div>

      <div className="settings-toolbar-filters">
        <button
          type="button"
          className={`filter-pill ${filters.modifiedOnly ? "active" : ""}`}
          onClick={() => toggleFilter("modifiedOnly")}
        >
          Modified
          {pendingCount > 0 && (
            <span className="filter-pill-count">{pendingCount}</span>
          )}
        </button>
        {mode === "pair" ? (
          <button
            type="button"
            className={`filter-pill ${filters.overriddenOnly ? "active" : ""}`}
            onClick={() => toggleFilter("overriddenOnly")}
          >
            Overridden
          </button>
        ) : null}
        <button
          type="button"
          className={`filter-pill ${filters.showAdvanced ? "active" : ""}`}
          onClick={() => toggleFilter("showAdvanced")}
        >
          Advanced
        </button>
        <button
          type="button"
          className={`filter-pill ${filters.requiresRestartOnly ? "active" : ""}`}
          onClick={() => toggleFilter("requiresRestartOnly")}
        >
          Restart required
        </button>
      </div>

      <div className="settings-toolbar-actions">
        {pendingCount > 0 ? (
          <button type="button" className="btn-ghost" onClick={onDiscard}>
            Discard
          </button>
        ) : null}
        {showSnapshotAction ? (
          <button
            type="button"
            className="btn-secondary"
            onClick={onSnapshot}
            title="Save current config as a named snapshot"
          >
            ⊡ Snapshot
          </button>
        ) : null}
        <button
          type="button"
          className={`btn-primary ${pendingCount === 0 ? "disabled" : ""}`}
          onClick={onSave}
          disabled={pendingCount === 0}
        >
          Save changes
          {pendingCount > 0 ? (
            <span className="btn-count">{pendingCount}</span>
          ) : null}
        </button>
      </div>
    </div>
  );
}
