"use client";

export type FilterState = {
  from: string;
  to: string;
  epic: string;
  outcome: string;
};

type Props = {
  filters: FilterState;
  onChange: (f: FilterState) => void;
  epics: string[];
};

const shortEpic = (epic: string) =>
  epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "");

export default function TradeFilters({ filters, onChange, epics }: Props) {
  const set = (key: keyof FilterState, value: string) =>
    onChange({ ...filters, [key]: value });

  return (
    <div className="ta-filters">
      <div className="ta-filter-group">
        <label>From</label>
        <input
          type="date"
          value={filters.from}
          onChange={(e) => set("from", e.target.value)}
          className="ta-input"
        />
      </div>
      <div className="ta-filter-group">
        <label>To</label>
        <input
          type="date"
          value={filters.to}
          onChange={(e) => set("to", e.target.value)}
          className="ta-input"
        />
      </div>
      <div className="ta-filter-group">
        <label>Pair</label>
        <select value={filters.epic} onChange={(e) => set("epic", e.target.value)} className="ta-input">
          <option value="">All pairs</option>
          {epics.map((e) => (
            <option key={e} value={e}>
              {shortEpic(e)}
            </option>
          ))}
        </select>
      </div>
      <div className="ta-filter-group">
        <label>Outcome</label>
        <select value={filters.outcome} onChange={(e) => set("outcome", e.target.value)} className="ta-input">
          <option value="">All</option>
          <option value="win">Win</option>
          <option value="loss">Loss</option>
          <option value="be">Breakeven</option>
        </select>
      </div>
      <button
        className="ta-clear-btn"
        onClick={() => onChange({ from: "", to: "", epic: "", outcome: "" })}
      >
        Clear
      </button>
    </div>
  );
}
