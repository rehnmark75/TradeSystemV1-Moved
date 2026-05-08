"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EnvironmentToggle from "../../../components/EnvironmentToggle";
import { useEnvironment } from "../../../lib/environment";
import { GOLD_EPIC } from "../../../lib/backtests";
import ForexNav from "../_components/ForexNav";
import type { BacktestsPayload, JobRow, LaunchFormState } from "./_lib/types";
import BacktestExecutionsTable from "./_components/BacktestExecutionsTable";
import BacktestFilters from "./_components/BacktestFilters";
import BacktestJobPanel from "./_components/BacktestJobPanel";
import BacktestLaunchForm from "./_components/BacktestLaunchForm";
import BacktestQueuedJobsTable from "./_components/BacktestQueuedJobsTable";
import BacktestSummaryCards from "./_components/BacktestSummaryCards";

const INITIAL_FORM: LaunchFormState = {
  epic: "CS.D.EURUSD.CEEM.IP",
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
  const [form, setForm] = useState<LaunchFormState>(INITIAL_FORM);

  const launchPairOptions = useMemo(() => {
    const options = payload?.form_options?.pairs ?? [];
    if (form.strategy === "XAU_GOLD") return options.filter((o) => o.value === GOLD_EPIC);
    return options.filter((o) => o.value !== GOLD_EPIC);
  }, [form.strategy, payload?.form_options?.pairs]);

  const loadBacktests = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ days: String(days), strategy, pair, limit: "20" });
    fetch(`/trading/api/forex/backtests/?${params}&env=${environment}`)
      .then((res) => res.json())
      .then((data) => {
        setPayload(data);
        if (!selectedJobId && data.jobs?.[0]?.job_id) setSelectedJobId(data.jobs[0].job_id);
      })
      .catch(() => setError("Failed to load backtests workspace."))
      .finally(() => setLoading(false));
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { loadBacktests(); }, [days, strategy, pair, environment]);

  useEffect(() => {
    if (!launchPairOptions.length) return;
    if (launchPairOptions.some((o) => o.value === form.epic)) return;
    setForm((f) => ({ ...f, epic: launchPairOptions[0].value }));
  }, [form.epic, launchPairOptions]);

  useEffect(() => {
    if (!selectedJobId) { setSelectedJob(null); return; }
    const controller = new AbortController();
    setJobStatusLoading(true);
    fetch(`/trading/api/forex/backtests/jobs/${selectedJobId}/`, { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => { if (!controller.signal.aborted) setSelectedJob(data.job ?? null); })
      .catch((err) => {
        if (!controller.signal.aborted && !(err instanceof Error && err.name === "AbortError")) {
          setSelectedJob(null);
        }
      })
      .finally(() => { if (!controller.signal.aborted) setJobStatusLoading(false); });
    return () => controller.abort();
  }, [selectedJobId]);

  useEffect(() => {
    if (!selectedJobId || !selectedJob) return;
    if (selectedJob.status !== "pending" && selectedJob.status !== "running") return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      fetch(`/trading/api/forex/backtests/jobs/${selectedJobId}/`, { signal: controller.signal })
        .then((res) => res.json())
        .then((data) => { if (!controller.signal.aborted) setSelectedJob(data.job ?? null); })
        .catch(() => undefined);
    }, 5000);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [selectedJob, selectedJobId]);

  const queueStats = useMemo(() => {
    const jobs = payload?.jobs ?? [];
    return {
      queued: jobs.filter((j) => j.status === "pending").length,
      running: jobs.filter((j) => j.status === "running").length,
      failed: jobs.filter((j) => j.status === "failed").length,
    };
  }, [payload?.jobs]);

  const executionStats = useMemo(() => {
    const rows = payload?.executions ?? [];
    const totalSignals = rows.reduce((s, r) => s + r.signal_count, 0);
    const totalWins = rows.reduce((s, r) => s + r.win_count, 0);
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
      if (!response.ok) throw new Error(data.error || "Failed to queue backtest");
      setSelectedJobId(data.job?.job_id ?? null);
      loadBacktests();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to queue backtest";
      if (message.toLowerCase().includes("variation")) setVariationError(message);
      else setSubmitError(message);
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
      if (!response.ok) throw new Error(data.error || `Failed to ${action} job`);
      if (action === "retry" && data.job?.job_id) setSelectedJobId(data.job.job_id);
      loadBacktests();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : `Failed to ${action} job`);
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="page backtests-page">
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
        <BacktestSummaryCards queueStats={queueStats} executionStats={executionStats} />

        <BacktestFilters
          days={days}
          strategy={strategy}
          pair={pair}
          filterOptions={{
            strategies: payload?.filters?.strategies ?? ["All"],
            epics: payload?.filters?.epics ?? ["All"],
          }}
          onDaysChange={setDays}
          onStrategyChange={setStrategy}
          onPairChange={setPair}
          onRefresh={loadBacktests}
        />

        {error ? <div className="error">{error}</div> : null}

        <div className="forex-grid">
          <BacktestLaunchForm
            form={form}
            formOptions={
              payload?.form_options ?? { pairs: [], strategies: [], timeframes: [], snapshots: [] }
            }
            environment={environment}
            submitting={submitting}
            submitError={submitError}
            variationError={variationError}
            launchPairOptions={launchPairOptions}
            onFormChange={(update) => setForm((f) => ({ ...f, ...update }))}
            onSubmit={submitJob}
            onRefreshWorkspace={loadBacktests}
          />

          <BacktestJobPanel
            selectedJob={selectedJob}
            loading={jobStatusLoading}
            actionLoading={actionLoading}
            onAction={performJobAction}
          />
        </div>

        <BacktestQueuedJobsTable
          jobs={payload?.jobs ?? []}
          loading={loading}
          onSelectJob={setSelectedJobId}
        />

        <BacktestExecutionsTable executions={payload?.executions ?? []} loading={loading} />
      </div>
    </div>
  );
}
