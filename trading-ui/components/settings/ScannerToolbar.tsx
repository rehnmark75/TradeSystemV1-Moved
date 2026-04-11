"use client";

interface ScannerFilterState {
  modifiedOnly: boolean;
  criticalOnly: boolean;
}

interface ScannerToolbarProps {
  query: string;
  onQueryChange: (q: string) => void;
  filters: ScannerFilterState;
  onFiltersChange: (filters: ScannerFilterState) => void;
  pendingCount: number;
  onSave: () => void;
  onDiscard: () => void;
}

export type { ScannerFilterState };

export default function ScannerToolbar({
  query,
  onQueryChange,
  filters,
  onFiltersChange,
  pendingCount,
  onSave,
  onDiscard,
}: ScannerToolbarProps) {
  const toggleFilter = (key: keyof ScannerFilterState) => {
    onFiltersChange({ ...filters, [key]: !filters[key] });
  };

  return (
    <div className="settings-toolbar">
      <div className="settings-toolbar-left">
        <div className="settings-toolbar-search">
          <span className="settings-toolbar-search-icon">⌕</span>
          <input
            type="text"
            placeholder="Search scanner parameters…"
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
        <button
          type="button"
          className={`filter-pill ${filters.criticalOnly ? "active" : ""}`}
          onClick={() => toggleFilter("criticalOnly")}
        >
          Critical
        </button>
      </div>

      <div className="settings-toolbar-actions">
        {pendingCount > 0 ? (
          <button type="button" className="btn-ghost" onClick={onDiscard}>
            Discard
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
