"use client";

import { GOLD_EPIC, getStrategyLabel } from "../../../../lib/backtests";
import type { LaunchFormState } from "../_lib/types";
import VariationGridEditor from "./VariationGridEditor";

type Props = {
  form: LaunchFormState;
  formOptions: {
    pairs: Array<{ label: string; value: string }>;
    strategies: string[];
    timeframes: string[];
    snapshots: Array<{ id: number; snapshot_name: string; description: string | null; created_at: string }>;
  };
  environment: string;
  submitting: boolean;
  submitError: string | null;
  variationError: string | null;
  launchPairOptions: Array<{ label: string; value: string }>;
  onFormChange: (update: Partial<LaunchFormState>) => void;
  onSubmit: () => void;
  onRefreshWorkspace: () => void;
};

export default function BacktestLaunchForm({
  form,
  formOptions,
  environment,
  submitting,
  submitError,
  variationError,
  launchPairOptions,
  onFormChange,
  onSubmit,
  onRefreshWorkspace,
}: Props) {
  return (
    <div className="panel table-panel backtest-launch-form">
      <div className="chart-title">Launch Backtest</div>

      <div className="backtest-launch-grid">
        <div>
          <label>Pair</label>
          <select value={form.epic} onChange={(e) => onFormChange({ epic: e.target.value })}>
            {launchPairOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Strategy</label>
          <select
            value={form.strategy}
            onChange={(e) =>
              onFormChange({
                strategy: e.target.value,
                epic:
                  e.target.value === "XAU_GOLD"
                    ? GOLD_EPIC
                    : form.epic === GOLD_EPIC
                      ? ""
                      : form.epic,
              })
            }
          >
            {(formOptions.strategies ?? []).map((opt) => (
              <option key={opt} value={opt}>
                {getStrategyLabel(opt)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Days</label>
          <input
            type="number"
            min={1}
            max={365}
            value={form.days}
            onChange={(e) => onFormChange({ days: Number(e.target.value || 14) })}
          />
        </div>
        <div>
          <label>Timeframe</label>
          <select value={form.timeframe} onChange={(e) => onFormChange({ timeframe: e.target.value })}>
            {(formOptions.timeframes ?? []).map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Start Date</label>
          <input
            type="date"
            value={form.start_date}
            onChange={(e) => onFormChange({ start_date: e.target.value })}
          />
        </div>
        <div>
          <label>End Date</label>
          <input
            type="date"
            value={form.end_date}
            onChange={(e) => onFormChange({ end_date: e.target.value })}
          />
        </div>
        <div>
          <label>Snapshot</label>
          <select
            value={form.snapshot_name}
            onChange={(e) => onFormChange({ snapshot_name: e.target.value })}
          >
            <option value="">Current active config</option>
            {(formOptions.snapshots ?? []).map((s) => (
              <option key={s.id} value={s.snapshot_name}>
                {s.snapshot_name}
              </option>
            ))}
          </select>
        </div>
        <div className="forex-badge">
          Environment
          <strong>{environment.toUpperCase()}</strong>
        </div>
      </div>

      <div className="backtest-toggle-grid">
        <label className="forex-badge backtest-toggle">
          Parallel
          <input
            type="checkbox"
            checked={form.parallel}
            onChange={(e) => onFormChange({ parallel: e.target.checked })}
          />
        </label>
        <label className="forex-badge backtest-toggle">
          Generate Chart
          <input
            type="checkbox"
            checked={form.generate_chart}
            onChange={(e) => onFormChange({ generate_chart: e.target.checked })}
          />
        </label>
        <label className="forex-badge backtest-toggle">
          Pipeline Mode
          <input
            type="checkbox"
            checked={form.pipeline_mode}
            onChange={(e) => onFormChange({ pipeline_mode: e.target.checked })}
          />
        </label>
        <label className="forex-badge backtest-toggle">
          Historical Intelligence
          <input
            type="checkbox"
            checked={form.use_historical_intelligence}
            onChange={(e) => onFormChange({ use_historical_intelligence: e.target.checked })}
          />
        </label>
      </div>

      {form.parallel ? (
        <div className="backtest-launch-grid">
          <div>
            <label>Workers</label>
            <input
              type="number"
              min={2}
              max={8}
              value={form.workers}
              onChange={(e) => onFormChange({ workers: Number(e.target.value || 4) })}
            />
          </div>
          <div>
            <label>Chunk Days</label>
            <input
              type="number"
              min={1}
              max={30}
              value={form.chunk_days}
              onChange={(e) => onFormChange({ chunk_days: Number(e.target.value || 7) })}
            />
          </div>
        </div>
      ) : null}

      <div className="panel table-panel backtest-variation-panel">
        <div className="chart-title">Parameter Variation</div>
        <label className="forex-badge backtest-toggle backtest-variation-toggle">
          Enable Variation Run
          <input
            type="checkbox"
            checked={form.variation_enabled}
            onChange={(e) => onFormChange({ variation_enabled: e.target.checked })}
          />
        </label>
        <VariationGridEditor
          enabled={form.variation_enabled}
          variationJson={form.variation_json}
          workers={form.variation_workers}
          rankBy={form.variation_rank_by}
          topN={form.variation_top_n}
          strategy={form.strategy}
          error={variationError}
          onChange={(payload) =>
            onFormChange({
              variation_json: payload.variationJson,
              variation_workers: payload.variationWorkers,
              variation_rank_by: payload.variationRankBy,
              variation_top_n: payload.variationTopN,
            })
          }
        />
      </div>

      {submitError ? <div className="error">{submitError}</div> : null}

      <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
        <button
          className="alert-history-button alert-history-button-active"
          disabled={submitting}
          onClick={onSubmit}
        >
          {submitting ? "Queueing..." : "Queue Backtest"}
        </button>
        <button className="alert-history-button" onClick={onRefreshWorkspace}>
          Refresh Workspace
        </button>
      </div>
    </div>
  );
}
