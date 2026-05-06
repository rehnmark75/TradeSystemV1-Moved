"use client";

import { useEffect, useRef, useState } from "react";
import { BacktestStrategy, STRATEGY_METADATA_ENDPOINT } from "../../../../lib/backtests";

type ParamMeta = {
  parameter_name: string;
  display_name: string;
  data_type: string;
  min_value: number | null;
  max_value: number | null;
  default_value: string | null;
  category: string;
  is_advanced: boolean;
};

type GridRow = {
  key: string;
  values: number[];
  inputText: string;
};

export type VariationChangePayload = {
  variationJson: string;
  variationWorkers: number;
  variationRankBy: string;
  variationTopN: number;
};

type Props = {
  enabled: boolean;
  variationJson: string;
  workers: number;
  rankBy: string;
  topN: number;
  strategy: string;
  error: string | null;
  onChange: (payload: VariationChangePayload) => void;
};

export default function VariationGridEditor({
  enabled,
  variationJson,
  workers,
  rankBy,
  topN,
  strategy,
  error,
  onChange,
}: Props) {
  const [mode, setMode] = useState<"typed" | "json">("typed");
  const [rows, setRows] = useState<GridRow[]>([]);
  const [metadata, setMetadata] = useState<ParamMeta[]>([]);
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaFallback, setMetaFallback] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const prevStrategyRef = useRef(strategy);

  useEffect(() => {
    if (strategy === prevStrategyRef.current) return;
    prevStrategyRef.current = strategy;
    setRows([]);
    setMetadata([]);
    setMetaFallback(false);
  }, [strategy]);

  useEffect(() => {
    const endpoint = STRATEGY_METADATA_ENDPOINT[strategy as BacktestStrategy];
    if (!endpoint) {
      setMetaFallback(true);
      return;
    }
    setMetaLoading(true);
    fetch(endpoint)
      .then((res) => {
        if (!res.ok) throw new Error("metadata fetch failed");
        return res.json() as Promise<ParamMeta[]>;
      })
      .then((data) => {
        const numeric = data.filter((p) =>
          ["integer", "numeric", "float", "number"].includes(p.data_type)
        );
        setMetadata(numeric);
        if (!numeric.length) setMetaFallback(true);
      })
      .catch(() => setMetaFallback(true))
      .finally(() => setMetaLoading(false));
  }, [strategy]);

  const visibleParams = showAdvanced ? metadata : metadata.filter((p) => !p.is_advanced);
  const usedKeys = new Set(rows.map((r) => r.key));
  const unusedParams = visibleParams.filter((p) => !usedKeys.has(p.parameter_name));
  const isAtMax = rows.length >= 5;
  const totalCombos = rows.reduce((acc, r) => acc * Math.max(r.values.length, 1), 1);
  const allRowsHaveValues = rows.length > 0 && rows.every((r) => r.values.length > 0);

  const serializeRows = (newRows: GridRow[]) => {
    const grid: Record<string, number[]> = {};
    for (const row of newRows) {
      if (row.values.length > 0) grid[row.key] = row.values;
    }
    return JSON.stringify(grid, null, 2);
  };

  const emit = (newRows: GridRow[]) => {
    onChange({
      variationJson: serializeRows(newRows),
      variationWorkers: workers,
      variationRankBy: rankBy,
      variationTopN: topN,
    });
  };

  const emitSettings = (patch: Partial<Omit<VariationChangePayload, "variationJson">>) => {
    onChange({
      variationJson,
      variationWorkers: workers,
      variationRankBy: rankBy,
      variationTopN: topN,
      ...patch,
    });
  };

  const addRow = () => {
    if (isAtMax || !unusedParams.length) return;
    const newRows = [...rows, { key: unusedParams[0].parameter_name, values: [], inputText: "" }];
    setRows(newRows);
    emit(newRows);
  };

  const removeRow = (index: number) => {
    const newRows = rows.filter((_, i) => i !== index);
    setRows(newRows);
    emit(newRows);
  };

  const changeRowKey = (index: number, key: string) => {
    const newRows = rows.map((r, i) => (i === index ? { ...r, key, values: [], inputText: "" } : r));
    setRows(newRows);
    emit(newRows);
  };

  const updateRowInput = (index: number, text: string) => {
    setRows(rows.map((r, i) => (i === index ? { ...r, inputText: text } : r)));
  };

  const addValue = (index: number) => {
    const row = rows[index];
    const num = parseFloat(row.inputText);
    if (!Number.isFinite(num)) return;
    if (row.values.length >= 10) return;
    const newRows = rows.map((r, i) =>
      i === index ? { ...r, values: [...r.values, num], inputText: "" } : r
    );
    setRows(newRows);
    emit(newRows);
  };

  const removeValue = (rowIndex: number, valIndex: number) => {
    const newRows = rows.map((r, i) =>
      i === rowIndex ? { ...r, values: r.values.filter((_, vi) => vi !== valIndex) } : r
    );
    setRows(newRows);
    emit(newRows);
  };

  if (!enabled) {
    return (
      <div className="chart-placeholder">
        Queue a single run or enable variation testing with a JSON grid.
      </div>
    );
  }

  const effectiveMode = metaFallback ? "json" : mode;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {!metaFallback && (
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button
            className={`alert-history-button${effectiveMode === "typed" ? " alert-history-button-active" : ""}`}
            onClick={() => setMode("typed")}
          >
            Typed
          </button>
          <button
            className={`alert-history-button${effectiveMode === "json" ? " alert-history-button-active" : ""}`}
            onClick={() => setMode("json")}
          >
            JSON
          </button>
        </div>
      )}
      {metaFallback && !metaLoading && (
        <div className="forex-badge" style={{ color: "#92400e" }}>
          Parameter metadata unavailable — using JSON editor
        </div>
      )}

      {effectiveMode === "typed" &&
        (metaLoading ? (
          <div className="chart-placeholder">Loading parameters...</div>
        ) : (
          <div style={{ display: "grid", gap: 8 }}>
            {rows.map((row, index) => {
              const meta = metadata.find((p) => p.parameter_name === row.key);
              const rangeText = meta
                ? [
                    meta.default_value != null && `Default: ${meta.default_value}`,
                    meta.min_value != null && meta.max_value != null &&
                      `Range: ${meta.min_value}–${meta.max_value}`,
                  ]
                    .filter(Boolean)
                    .join("  ")
                : "";
              return (
                <div
                  key={index}
                  className="panel table-panel"
                  style={{ padding: 12, display: "grid", gap: 8 }}
                >
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <select
                      value={row.key}
                      onChange={(e) => changeRowKey(index, e.target.value)}
                      style={{ flex: 1 }}
                    >
                      <option value={row.key}>{meta?.display_name ?? row.key}</option>
                      {unusedParams
                        .filter((p) => p.parameter_name !== row.key)
                        .map((p) => (
                          <option key={p.parameter_name} value={p.parameter_name}>
                            {p.display_name}
                          </option>
                        ))}
                    </select>
                    {rangeText && (
                      <span style={{ fontSize: 12, color: "#6b7280", whiteSpace: "nowrap" }}>
                        {rangeText}
                      </span>
                    )}
                    <button className="alert-history-button" onClick={() => removeRow(index)}>
                      ×
                    </button>
                  </div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    {row.values.map((v, vi) => (
                      <span
                        key={vi}
                        className="forex-badge"
                        style={{ cursor: "pointer" }}
                        onClick={() => removeValue(index, vi)}
                        title="Click to remove"
                      >
                        {v} ×
                      </span>
                    ))}
                    <input
                      type="number"
                      value={row.inputText}
                      onChange={(e) => updateRowInput(index, e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") addValue(index);
                      }}
                      placeholder="add value"
                      style={{ width: 100 }}
                      disabled={row.values.length >= 10}
                    />
                    <button
                      className="alert-history-button"
                      onClick={() => addValue(index)}
                      disabled={row.values.length >= 10}
                    >
                      + value
                    </button>
                  </div>
                </div>
              );
            })}

            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <button
                className="alert-history-button alert-history-button-active"
                onClick={addRow}
                disabled={isAtMax || !unusedParams.length}
              >
                + Add Parameter
              </button>
              {metadata.some((p) => p.is_advanced) && (
                <label className="forex-badge" style={{ gap: 8 }}>
                  Show advanced
                  <input
                    type="checkbox"
                    checked={showAdvanced}
                    onChange={(e) => setShowAdvanced(e.target.checked)}
                  />
                </label>
              )}
              {allRowsHaveValues && (
                <span style={{ fontSize: 12, color: "#6b7280" }}>
                  Combinations: {totalCombos}
                </span>
              )}
            </div>
          </div>
        ))}

      {effectiveMode === "json" && (
        <>
          <textarea
            value={variationJson}
            onChange={(e) =>
              onChange({
                variationJson: e.target.value,
                variationWorkers: workers,
                variationRankBy: rankBy,
                variationTopN: topN,
              })
            }
            placeholder={
              '{\n  "min_confidence": [0.45, 0.5, 0.55],\n  "fixed_stop_loss_pips": [8, 10, 12],\n  "fixed_take_profit_pips": [10, 12, 15]\n}'
            }
            rows={8}
            style={{ width: "100%" }}
          />
          <details className="forex-badge" style={{ display: "block" }}>
            <summary>Common variation keys</summary>
            <div style={{ marginTop: 8 }}>
              min_confidence, fixed_stop_loss_pips, fixed_take_profit_pips, min_rr_ratio,
              sl_buffer_pips, ema_period, cooldown_minutes
            </div>
          </details>
        </>
      )}

      <div className="forex-controls" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
        <div>
          <label>Variation Workers</label>
          <input
            type="number"
            min={2}
            max={8}
            value={workers}
            onChange={(e) => emitSettings({ variationWorkers: Number(e.target.value || 4) })}
          />
        </div>
        <div>
          <label>Rank By</label>
          <select value={rankBy} onChange={(e) => emitSettings({ variationRankBy: e.target.value })}>
            <option value="composite_score">Composite Score</option>
            <option value="win_rate">Win Rate</option>
            <option value="total_pips">Total Pips</option>
            <option value="profit_factor">Profit Factor</option>
            <option value="expectancy">Expectancy</option>
          </select>
        </div>
        <div>
          <label>Top N</label>
          <input
            type="number"
            min={1}
            max={50}
            value={topN}
            onChange={(e) => emitSettings({ variationTopN: Number(e.target.value || 10) })}
          />
        </div>
      </div>
      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}
