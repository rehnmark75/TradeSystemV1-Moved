"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type JobRow = {
  id: number;
  job_id: string;
  status: string;
  epic: string;
  days: number;
  strategy: string;
  timeframe: string;
  parallel: boolean;
  workers: number | null;
  chunk_days: number | null;
  generate_chart: boolean;
  pipeline_mode: boolean;
  snapshot_name: string | null;
  use_historical_intelligence?: boolean;
  variation_config?: {
    enabled?: boolean;
    param_grid?: Record<string, unknown[]>;
    workers?: number;
    rank_by?: string;
    top_n?: number;
  } | null;
  progress?: {
    phase?: string;
    elapsed_seconds?: number;
    last_activity?: string | null;
    current?: number;
    total?: number;
  } | null;
  recent_output?: string[] | null;
  cancel_requested_at?: string | null;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  execution_id: number | null;
  error_message: string | null;
};

type ExecutionRow = {
  id: number;
  strategy_name: string | null;
  start_time: string;
  status: string;
  epics_tested: string[] | null;
  execution_duration_seconds: number | null;
  chart_url: string | null;
  signal_count: number;
  win_count: number;
  loss_count: number;
  total_pips: number;
  win_rate: number;
};

type BacktestsPayload = {
  filters: {
    days: number;
    strategy: string;
    pair: string;
    strategies: string[];
    epics: string[];
  };
  form_options: {
    pairs: Array<{ label: string; value: string }>;
    strategies: string[];
    timeframes: string[];
    snapshots: Array<{ id: number; snapshot_name: string; description: string | null; created_at: string }>;
  };
  jobs: JobRow[];
  executions: ExecutionRow[];
};

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const formatDateTime = (value: string | null | undefined) => {
  if (!value) return "N/A";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatDuration = (seconds: number | null | undefined) => {
  if (!seconds || seconds <= 0) return "Pending";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
};

const epicToPair = (epic: string | null | undefined) => {
  if (!epic) return "N/A";
  const parts = epic.split(".");
  if (parts.length >= 3) return parts[2].slice(0, 6);
  return epic;
};

export default function ForexBacktestsPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(14);
  const [strategy, setStrategy] = useState("All");
  const [pair, setPair] = useState("All");
  const [payload, setPayload] = useState<BacktestsPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [variationError, setVariationError] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRow | null>(null);
  const [jobStatusLoading, setJobStatusLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const [form, setForm] = useState({
    epic: "CS.D.EURUSD.MINI.IP",
    days: 14,
    strategy: "SMC_SIMPLE",
    timeframe: "15m",
    parallel: false,
    workers: 4,
    chunk_days: 7,
    generate_chart: true,
    pipeline_mode: false,
    use_historical_intelligence: false,
    start_date: "",
    end_date: "",
    snapshot_name: "",
    variation_enabled: false,
    variation_json: '{\n  "fixed_stop_loss_pips": [8, 10, 12],\n  "min_confidence": [0.45, 0.5, 0.55]\n}',
    variation_workers: 4,
    variation_rank_by: "composite_score",
    variation_top_n: 10,
  });

  const loadBacktests = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      days: String(days),
      strategy,
      pair,
      limit: "20",
    });

    fetch(`/trading/api/forex/backtests/?${params.toString()}&env=${environment}`)
      .then((res) => res.json())
      .then((data) => {
        setPayload(data);
        if (!selectedJobId && data.jobs?.[0]?.job_id) {
          setSelectedJobId(data.jobs[0].job_id);
        }
      })
      .catch(() => setError("Failed to load backtests workspace."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadBacktests();
  }, [days, strategy, pair, environment]);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJob(null);
      return;
    }

    setJobStatusLoading(true);
    fetch(`/trading/api/forex/backtests/jobs/${selectedJobId}/`)
      .then((res) => res.json())
      .then((data) => setSelectedJob(data.job ?? null))
      .catch(() => setSelectedJob(null))
      .finally(() => setJobStatusLoading(false));
  }, [selectedJobId]);

  useEffect(() => {
    if (!selectedJobId || !selectedJob) return;
    if (selectedJob.status !== "pending" && selectedJob.status !== "running") return;

    const timer = window.setTimeout(() => {
      fetch(`/trading/api/forex/backtests/jobs/${selectedJobId}/`)
        .then((res) => res.json())
        .then((data) => setSelectedJob(data.job ?? null))
        .catch(() => undefined);
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [selectedJob, selectedJobId]);

  const queueStats = useMemo(() => {
    const jobs = payload?.jobs ?? [];
    return {
      queued: jobs.filter((job) => job.status === "pending").length,
      running: jobs.filter((job) => job.status === "running").length,
      failed: jobs.filter((job) => job.status === "failed").length,
    };
  }, [payload?.jobs]);

  const executionStats = useMemo(() => {
    const rows = payload?.executions ?? [];
    const totalSignals = rows.reduce((sum, row) => sum + row.signal_count, 0);
    const totalWins = rows.reduce((sum, row) => sum + row.win_count, 0);
    return {
      executions: rows.length,
      totalSignals,
      avgWinRate: totalSignals > 0 ? (totalWins / totalSignals) * 100 : 0,
    };
  }, [payload?.executions]);

  const submitJob = async () => {
    setSubmitting(true);
    setSubmitError(null);
    setVariationError(null);

    try {
      let variationConfig: Record<string, unknown> | null = null;
      if (form.variation_enabled) {
        let parsedGrid: unknown;
        try {
          parsedGrid = JSON.parse(form.variation_json);
        } catch {
          throw new Error("Variation JSON is invalid");
        }
        variationConfig = {
          enabled: true,
          param_grid: parsedGrid,
          workers: form.variation_workers,
          rank_by: form.variation_rank_by,
          top_n: form.variation_top_n,
        };
      }

      const response = await fetch("/trading/api/forex/backtests/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          submitted_by: "trading-ui",
          workers: form.parallel ? form.workers : null,
          chunk_days: form.parallel ? form.chunk_days : null,
          variation_config: variationConfig,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failed to queue backtest");
      }

      setSelectedJobId(data.job?.job_id ?? null);
      loadBacktests();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to queue backtest";
      if (message.toLowerCase().includes("variation")) {
        setVariationError(message);
      } else {
        setSubmitError(message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const performJobAction = async (jobId: string, action: "cancel" | "retry") => {
    setActionLoading(`${action}:${jobId}`);
    setSubmitError(null);
    try {
      const response = await fetch(`/trading/api/forex/backtests/jobs/${jobId}/${action}/`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Failed to ${action} job`);
      }
      if (action === "retry" && data.job?.job_id) {
        setSelectedJobId(data.job.job_id);
      }
      loadBacktests();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : `Failed to ${action} job`);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          K.L.I.R.R
        </Link>
        <EnvironmentToggle />
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Backtests</h1>
          <p>Queue forex backtests, monitor worker-side execution, and review recent results without Streamlit.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/backtests" />

      <div className="panel">
        <div className="metrics-grid">
          <div className="summary-card">
            Queued Jobs
            <strong>{queueStats.queued}</strong>
          </div>
          <div className="summary-card">
            Running Jobs
            <strong>{queueStats.running}</strong>
          </div>
          <div className="summary-card">
            Recent Executions
            <strong>{executionStats.executions}</strong>
          </div>
          <div className="summary-card">
            Avg Win Rate
            <strong>{executionStats.avgWinRate.toFixed(1)}%</strong>
          </div>
        </div>

        <div className="forex-controls">
          <div>
            <label>Window</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Strategy</label>
            <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
              {(payload?.filters?.strategies ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Pair</label>
            <select value={pair} onChange={(event) => setPair(event.target.value)}>
              {(payload?.filters?.epics ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option === "All" ? option : epicToPair(option)}
                </option>
              ))}
            </select>
          </div>
          <button className="alert-history-button alert-history-button-active" onClick={loadBacktests}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="forex-grid">
          <div className="panel table-panel">
            <div className="chart-title">Launch Backtest</div>
            <div className="forex-controls" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
              <div>
                <label>Pair</label>
                <select
                  value={form.epic}
                  onChange={(event) => setForm((current) => ({ ...current, epic: event.target.value }))}
                >
                  {(payload?.form_options?.pairs ?? []).map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Strategy</label>
                <select
                  value={form.strategy}
                  onChange={(event) => setForm((current) => ({ ...current, strategy: event.target.value }))}
                >
                  {(payload?.form_options?.strategies ?? []).map((option) => (
                    <option key={option} value={option}>
                      {option}
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
                  onChange={(event) =>
                    setForm((current) => ({ ...current, days: Number(event.target.value || 14) }))
                  }
                />
              </div>
              <div>
                <label>Timeframe</label>
                <select
                  value={form.timeframe}
                  onChange={(event) => setForm((current) => ({ ...current, timeframe: event.target.value }))}
                >
                  {(payload?.form_options?.timeframes ?? []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Start Date</label>
                <input
                  type="date"
                  value={form.start_date}
                  onChange={(event) => setForm((current) => ({ ...current, start_date: event.target.value }))}
                />
              </div>
              <div>
                <label>End Date</label>
                <input
                  type="date"
                  value={form.end_date}
                  onChange={(event) => setForm((current) => ({ ...current, end_date: event.target.value }))}
                />
              </div>
              <div>
                <label>Snapshot</label>
                <select
                  value={form.snapshot_name}
                  onChange={(event) => setForm((current) => ({ ...current, snapshot_name: event.target.value }))}
                >
                  <option value="">Current active config</option>
                  {(payload?.form_options?.snapshots ?? []).map((snapshot) => (
                    <option key={snapshot.id} value={snapshot.snapshot_name}>
                      {snapshot.snapshot_name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="forex-badge">
                Environment
                <strong>{environment.toUpperCase()}</strong>
              </div>
            </div>

            <div className="forex-controls" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
              <label className="forex-badge" style={{ justifyContent: "space-between" }}>
                Parallel
                <input
                  type="checkbox"
                  checked={form.parallel}
                  onChange={(event) => setForm((current) => ({ ...current, parallel: event.target.checked }))}
                />
              </label>
              <label className="forex-badge" style={{ justifyContent: "space-between" }}>
                Generate Chart
                <input
                  type="checkbox"
                  checked={form.generate_chart}
                  onChange={(event) =>
                    setForm((current) => ({ ...current, generate_chart: event.target.checked }))
                  }
                />
              </label>
              <label className="forex-badge" style={{ justifyContent: "space-between" }}>
                Pipeline Mode
                <input
                  type="checkbox"
                  checked={form.pipeline_mode}
                  onChange={(event) =>
                    setForm((current) => ({ ...current, pipeline_mode: event.target.checked }))
                  }
                />
              </label>
              <label className="forex-badge" style={{ justifyContent: "space-between" }}>
                Historical Intelligence
                <input
                  type="checkbox"
                  checked={form.use_historical_intelligence}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      use_historical_intelligence: event.target.checked,
                    }))
                  }
                />
              </label>
            </div>

            {form.parallel ? (
              <div className="forex-controls" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
                <div>
                  <label>Workers</label>
                  <input
                    type="number"
                    min={2}
                    max={8}
                    value={form.workers}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, workers: Number(event.target.value || 4) }))
                    }
                  />
                </div>
                <div>
                  <label>Chunk Days</label>
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={form.chunk_days}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, chunk_days: Number(event.target.value || 7) }))
                    }
                  />
                </div>
              </div>
            ) : null}

            <div className="panel table-panel" style={{ marginTop: 16 }}>
              <div className="chart-title">Parameter Variation</div>
              <label className="forex-badge" style={{ justifyContent: "space-between", marginBottom: 12 }}>
                Enable Variation Run
                <input
                  type="checkbox"
                  checked={form.variation_enabled}
                  onChange={(event) =>
                    setForm((current) => ({ ...current, variation_enabled: event.target.checked }))
                  }
                />
              </label>

              {form.variation_enabled ? (
                <div style={{ display: "grid", gap: 12 }}>
                  <textarea
                    value={form.variation_json}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, variation_json: event.target.value }))
                    }
                    rows={8}
                    style={{ width: "100%" }}
                  />
                  <div className="forex-controls" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
                    <div>
                      <label>Variation Workers</label>
                      <input
                        type="number"
                        min={2}
                        max={8}
                        value={form.variation_workers}
                        onChange={(event) =>
                          setForm((current) => ({
                            ...current,
                            variation_workers: Number(event.target.value || 4),
                          }))
                        }
                      />
                    </div>
                    <div>
                      <label>Rank By</label>
                      <select
                        value={form.variation_rank_by}
                        onChange={(event) =>
                          setForm((current) => ({ ...current, variation_rank_by: event.target.value }))
                        }
                      >
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
                        value={form.variation_top_n}
                        onChange={(event) =>
                          setForm((current) => ({
                            ...current,
                            variation_top_n: Number(event.target.value || 10),
                          }))
                        }
                      />
                    </div>
                  </div>
                  {variationError ? <div className="error">{variationError}</div> : null}
                </div>
              ) : (
                <div className="chart-placeholder">Queue a single run or enable variation testing with a JSON grid.</div>
              )}
            </div>

            {submitError ? <div className="error">{submitError}</div> : null}

            <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
              <button
                className="alert-history-button alert-history-button-active"
                disabled={submitting}
                onClick={submitJob}
              >
                {submitting ? "Queueing..." : "Queue Backtest"}
              </button>
              <button className="alert-history-button" onClick={loadBacktests}>
                Refresh Workspace
              </button>
            </div>
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Selected Job</div>
            {jobStatusLoading ? <div className="chart-placeholder">Loading job...</div> : null}
            {!jobStatusLoading && !selectedJob ? (
              <div className="chart-placeholder">Select a queued job to inspect status.</div>
            ) : null}
            {!jobStatusLoading && selectedJob ? (
              <div style={{ display: "grid", gap: 12 }}>
                <div className="forex-badge">
                  {selectedJob.status.toUpperCase()}
                  <strong>{epicToPair(selectedJob.epic)}</strong>
                </div>
                <div className="metrics-grid">
                  <div className="summary-card">
                    Job ID
                    <strong>{selectedJob.job_id}</strong>
                  </div>
                  <div className="summary-card">
                    Timeframe
                    <strong>{selectedJob.timeframe}</strong>
                  </div>
                  <div className="summary-card">
                    Submitted
                    <strong>{formatDateTime(selectedJob.submitted_at)}</strong>
                  </div>
                  <div className="summary-card">
                    Finished
                    <strong>{formatDateTime(selectedJob.completed_at)}</strong>
                  </div>
                </div>
                <div className="forex-badge">
                  Config
                  <strong>
                    {selectedJob.days}d {selectedJob.strategy} {selectedJob.parallel ? "Parallel" : "Single"}
                  </strong>
                </div>
                {selectedJob.snapshot_name ? (
                  <div className="forex-badge">
                    Snapshot
                    <strong>{selectedJob.snapshot_name}</strong>
                  </div>
                ) : null}
                {selectedJob.use_historical_intelligence ? (
                  <div className="forex-badge">
                    Historical Intelligence
                    <strong>Enabled</strong>
                  </div>
                ) : null}
                {selectedJob.variation_config?.enabled ? (
                  <div className="forex-badge">
                    Variation Run
                    <strong>{Object.keys(selectedJob.variation_config.param_grid ?? {}).length} parameters</strong>
                  </div>
                ) : null}
                {selectedJob.progress ? (
                  <div className="panel table-panel">
                    <div className="chart-title">Live Progress</div>
                    <table className="forex-table">
                      <tbody>
                        <tr>
                          <td>Phase</td>
                          <td>{selectedJob.progress.phase ?? "running"}</td>
                        </tr>
                        <tr>
                          <td>Elapsed</td>
                          <td>
                            {selectedJob.progress.elapsed_seconds != null
                              ? `${Math.round(selectedJob.progress.elapsed_seconds)}s`
                              : "N/A"}
                          </td>
                        </tr>
                        <tr>
                          <td>Activity</td>
                          <td>{selectedJob.progress.last_activity ?? "N/A"}</td>
                        </tr>
                        {selectedJob.progress.current && selectedJob.progress.total ? (
                          <tr>
                            <td>Variation Progress</td>
                            <td>
                              {selectedJob.progress.current}/{selectedJob.progress.total}
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                    {selectedJob.recent_output?.length ? (
                      <pre style={{ whiteSpace: "pre-wrap", marginTop: 12 }}>
                        {selectedJob.recent_output.slice(-10).join("\n")}
                      </pre>
                    ) : null}
                  </div>
                ) : null}
                {selectedJob.execution_id ? (
                  <Link
                    href={`/forex/backtests/${selectedJob.execution_id}`}
                    className="alert-history-button alert-history-button-active"
                  >
                    Open Execution #{selectedJob.execution_id}
                  </Link>
                ) : null}
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {selectedJob.status === "pending" || selectedJob.status === "running" ? (
                    <button
                      className="alert-history-button"
                      disabled={
                        actionLoading === `cancel:${selectedJob.job_id}` ||
                        Boolean(selectedJob.cancel_requested_at)
                      }
                      onClick={() => performJobAction(selectedJob.job_id, "cancel")}
                    >
                      {actionLoading === `cancel:${selectedJob.job_id}` || selectedJob.cancel_requested_at
                        ? "Cancelling..."
                        : selectedJob.status === "running"
                          ? "Cancel Running Job"
                          : "Cancel Pending Job"}
                    </button>
                  ) : null}
                  {selectedJob.status === "failed" || selectedJob.status === "completed" || selectedJob.status === "cancelled" ? (
                    <button
                      className="alert-history-button"
                      disabled={actionLoading === `retry:${selectedJob.job_id}`}
                      onClick={() => performJobAction(selectedJob.job_id, "retry")}
                    >
                      {actionLoading === `retry:${selectedJob.job_id}` ? "Retrying..." : "Retry Job"}
                    </button>
                  ) : null}
                </div>
                {selectedJob.error_message ? <div className="error">{selectedJob.error_message}</div> : null}
              </div>
            ) : null}
          </div>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Queued Jobs</div>
          {loading ? (
            <div className="chart-placeholder">Loading jobs...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Pair</th>
                  <th>Status</th>
                  <th>Strategy</th>
                  <th>Submitted</th>
                  <th>Execution</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.jobs ?? []).map((job) => (
                  <tr key={job.job_id} onClick={() => setSelectedJobId(job.job_id)} style={{ cursor: "pointer" }}>
                    <td>{job.job_id}</td>
                    <td>{epicToPair(job.epic)}</td>
                    <td>{job.status}</td>
                    <td>
                      {job.strategy} · {job.timeframe}
                    </td>
                    <td>{formatDateTime(job.submitted_at)}</td>
                    <td>{job.execution_id ? `#${job.execution_id}` : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Recent Executions</div>
          {loading ? (
            <div className="chart-placeholder">Loading executions...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Started</th>
                  <th>Pair</th>
                  <th>Strategy</th>
                  <th>Signals</th>
                  <th>Win Rate</th>
                  <th>Total Pips</th>
                  <th>Duration</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.executions ?? []).map((execution) => (
                  <tr key={execution.id}>
                    <td>{execution.id}</td>
                    <td>{formatDateTime(execution.start_time)}</td>
                    <td>{epicToPair(execution.epics_tested?.[0])}</td>
                    <td>{execution.strategy_name ?? "N/A"}</td>
                    <td>{execution.signal_count}</td>
                    <td>{execution.win_rate.toFixed(1)}%</td>
                    <td>{execution.total_pips.toFixed(1)}</td>
                    <td>{formatDuration(execution.execution_duration_seconds)}</td>
                    <td>
                      <Link href={`/forex/backtests/${execution.id}`}>Open</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
