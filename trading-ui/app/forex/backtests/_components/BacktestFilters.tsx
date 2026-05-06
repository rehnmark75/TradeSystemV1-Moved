"use client";

import { epicToPair } from "../../../../lib/backtests";

const DAY_OPTIONS = [7, 14, 30, 60, 90];

type Props = {
  days: number;
  strategy: string;
  pair: string;
  filterOptions: { strategies: string[]; epics: string[] };
  onDaysChange: (days: number) => void;
  onStrategyChange: (strategy: string) => void;
  onPairChange: (pair: string) => void;
  onRefresh: () => void;
};

export default function BacktestFilters({
  days,
  strategy,
  pair,
  filterOptions,
  onDaysChange,
  onStrategyChange,
  onPairChange,
  onRefresh,
}: Props) {
  return (
    <div className="forex-controls">
      <div>
        <label>Window</label>
        <select value={days} onChange={(e) => onDaysChange(Number(e.target.value))}>
          {DAY_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}d
            </option>
          ))}
        </select>
      </div>
      <div>
        <label>Strategy</label>
        <select value={strategy} onChange={(e) => onStrategyChange(e.target.value)}>
          {(filterOptions.strategies ?? ["All"]).map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label>Pair</label>
        <select value={pair} onChange={(e) => onPairChange(e.target.value)}>
          {(filterOptions.epics ?? ["All"]).map((opt) => (
            <option key={opt} value={opt}>
              {opt === "All" ? opt : epicToPair(opt)}
            </option>
          ))}
        </select>
      </div>
      <button className="alert-history-button alert-history-button-active" onClick={onRefresh}>
        Refresh
      </button>
    </div>
  );
}
