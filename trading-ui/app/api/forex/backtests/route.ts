import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";
import {
  BACKTEST_STRATEGIES,
  BACKTEST_TIMEFRAMES,
  DEFAULT_BACKTEST_DAYS,
  DEFAULT_BACKTEST_LIMIT,
  coerceBoolean,
  getEpicOptions,
  isRecord,
  parsePositiveInt,
  validateVariationConfig,
} from "../../../../lib/backtests";

export const dynamic = "force-dynamic";

type BacktestJobInsert = {
  epic: string;
  days: number;
  strategy: string;
  timeframe: string;
  parallel: boolean;
  workers: number | null;
  chunk_days: number | null;
  generate_chart: boolean;
  pipeline_mode: boolean;
  start_date: string | null;
  end_date: string | null;
  snapshot_name: string | null;
  use_historical_intelligence: boolean;
  variation_config: Record<string, unknown> | null;
  parameter_overrides: Record<string, unknown>;
  submitted_by: string;
};

function parseJobBody(body: unknown): BacktestJobInsert | { error: string; status: number } {
  if (!isRecord(body)) {
    return { error: "Invalid request body", status: 400 };
  }

  const epic = typeof body.epic === "string" ? body.epic.trim() : "";
  const strategy = typeof body.strategy === "string" ? body.strategy.trim() : "";
  const timeframe = typeof body.timeframe === "string" ? body.timeframe.trim() : "";
  const snapshotName = typeof body.snapshot_name === "string" ? body.snapshot_name.trim() : "";
  const submittedBy = typeof body.submitted_by === "string" ? body.submitted_by.trim() : "trading-ui";
  const startDate = typeof body.start_date === "string" ? body.start_date.trim() : "";
  const endDate = typeof body.end_date === "string" ? body.end_date.trim() : "";
  const variationValidation = validateVariationConfig(body.variation_config);
  if (variationValidation.error) {
    return { error: variationValidation.error, status: 400 };
  }

  if (!epic) return { error: "Epic is required", status: 400 };
  if (!BACKTEST_STRATEGIES.includes(strategy as (typeof BACKTEST_STRATEGIES)[number])) {
    return { error: "Unsupported strategy", status: 400 };
  }
  if (!BACKTEST_TIMEFRAMES.includes(timeframe as (typeof BACKTEST_TIMEFRAMES)[number])) {
    return { error: "Unsupported timeframe", status: 400 };
  }

  const rawDays = Number(body.days);
  const days = Number.isFinite(rawDays) && rawDays > 0 ? Math.floor(rawDays) : DEFAULT_BACKTEST_DAYS;
  const parallel = coerceBoolean(body.parallel);
  const generateChart = body.generate_chart == null ? true : coerceBoolean(body.generate_chart);
  const pipelineMode = coerceBoolean(body.pipeline_mode);
  const workersRaw = Number(body.workers);
  const chunkDaysRaw = Number(body.chunk_days);
  const workers = parallel && Number.isFinite(workersRaw) && workersRaw >= 2 ? Math.floor(workersRaw) : null;
  const chunkDays =
    parallel && Number.isFinite(chunkDaysRaw) && chunkDaysRaw >= 1 ? Math.floor(chunkDaysRaw) : null;
  const parameterOverrides = isRecord(body.parameter_overrides) ? body.parameter_overrides : {};

  return {
    epic,
    days,
    strategy,
    timeframe,
    parallel,
    workers,
    chunk_days: chunkDays,
    generate_chart: generateChart,
    pipeline_mode: pipelineMode,
    start_date: startDate || null,
    end_date: endDate || null,
    snapshot_name: snapshotName || null,
    use_historical_intelligence: coerceBoolean(body.use_historical_intelligence),
    variation_config: variationValidation.value,
    parameter_overrides: parameterOverrides,
    submitted_by: submittedBy || "trading-ui",
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parsePositiveInt(searchParams.get("days"), DEFAULT_BACKTEST_DAYS);
  const limit = parsePositiveInt(searchParams.get("limit"), DEFAULT_BACKTEST_LIMIT);
  const strategy = searchParams.get("strategy") || "All";
  const pair = searchParams.get("pair") || "All";
  const configSet = searchParams.get("env") || "demo";
  const since = new Date();
  since.setDate(since.getDate() - days);

  const params: unknown[] = [since];
  const executionFilters = ["be.start_time >= $1"];
  const jobFilters = ["submitted_at >= $1"];

  if (strategy !== "All") {
    params.push(strategy);
    executionFilters.push(`be.strategy_name = $${params.length}`);
    jobFilters.push(`strategy = $${params.length}`);
  }

  if (pair !== "All") {
    params.push(pair);
    executionFilters.push(`EXISTS (
      SELECT 1 FROM unnest(be.epics_tested) AS epic
      WHERE epic = $${params.length}
    )`);
    jobFilters.push(`epic = $${params.length}`);
  }

  params.push(limit);
  const limitParam = `$${params.length}`;

  try {
    const [executionsResult, jobsResult, strategyResult, epicResult, snapshotResult] = await Promise.all([
      forexPool.query(
        `
        SELECT
          be.id,
          be.execution_name,
          be.strategy_name,
          be.start_time,
          be.end_time,
          be.data_start_date,
          be.data_end_date,
          be.epics_tested,
          be.status,
          be.execution_duration_seconds,
          be.chart_url,
          COALESCE(bs.signal_count, 0) AS signal_count,
          COALESCE(bs.win_count, 0) AS win_count,
          COALESCE(bs.loss_count, 0) AS loss_count,
          COALESCE(bs.total_pips, 0) AS total_pips
        FROM backtest_executions be
        LEFT JOIN (
          SELECT
            execution_id,
            COUNT(*) AS signal_count,
            SUM(CASE WHEN pips_gained > 0 THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN pips_gained <= 0 THEN 1 ELSE 0 END) AS loss_count,
            COALESCE(SUM(pips_gained), 0) AS total_pips
          FROM backtest_signals
          GROUP BY execution_id
        ) bs ON bs.execution_id = be.id
        WHERE ${executionFilters.join(" AND ")}
        ORDER BY be.start_time DESC
        LIMIT ${limitParam}
        `,
        params
      ),
      forexPool.query(
        `
        SELECT
          id,
          job_id,
          status,
          epic,
          days,
          strategy,
          timeframe,
          parallel,
          workers,
          chunk_days,
          generate_chart,
          pipeline_mode,
          parameter_overrides,
          snapshot_name,
          use_historical_intelligence,
          variation_config,
          submitted_at,
          started_at,
          completed_at,
          execution_id,
          error_message,
          submitted_by
        FROM backtest_job_queue
        WHERE ${jobFilters.join(" AND ")}
        ORDER BY submitted_at DESC
        LIMIT ${limitParam}
        `,
        params
      ),
      forexPool.query(
        `
        SELECT DISTINCT strategy_name
        FROM backtest_executions
        WHERE strategy_name IS NOT NULL
        ORDER BY strategy_name
        `
      ),
      forexPool.query(
        `
        SELECT DISTINCT unnest(epics_tested) AS epic
        FROM backtest_executions
        WHERE epics_tested IS NOT NULL
        ORDER BY epic
        `
      ),
      strategyConfigPool.query(
        `
        SELECT id, snapshot_name, description, created_at
        FROM smc_backtest_snapshots
        WHERE is_active = TRUE
          AND is_backtest_only = FALSE
          AND (
            $1::text IS NULL
            OR COALESCE(parameter_overrides->>'config_set', 'demo') = $1
          )
        ORDER BY created_at DESC
        LIMIT 50
        `,
        [configSet]
      ),
    ]);

    const executionRows = executionsResult.rows.map((row) => {
      const signalCount = Number(row.signal_count ?? 0);
      const winCount = Number(row.win_count ?? 0);
      return {
        ...row,
        signal_count: signalCount,
        win_count: winCount,
        loss_count: Number(row.loss_count ?? 0),
        total_pips: Number(row.total_pips ?? 0),
        win_rate: signalCount > 0 ? (winCount / signalCount) * 100 : 0,
      };
    });

    return NextResponse.json({
      filters: {
        days,
        strategy,
        pair,
        strategies: ["All", ...strategyResult.rows.map((row) => row.strategy_name)],
        epics: ["All", ...epicResult.rows.map((row) => row.epic)],
      },
      form_options: {
        pairs: getEpicOptions(),
        strategies: [...BACKTEST_STRATEGIES],
        timeframes: [...BACKTEST_TIMEFRAMES],
        snapshots: snapshotResult.rows,
      },
      jobs: jobsResult.rows,
      executions: executionRows,
    });
  } catch (error) {
    console.error("Failed to load forex backtests", error);
    return NextResponse.json({ error: "Failed to load forex backtests" }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const parsed = parseJobBody(body);
  if ("error" in parsed) {
    return NextResponse.json({ error: parsed.error }, { status: parsed.status });
  }

  try {
    const insertResult = await forexPool.query(
      `
      INSERT INTO backtest_job_queue (
        job_id,
        status,
        epic,
        days,
        strategy,
        timeframe,
        parallel,
        workers,
        chunk_days,
        generate_chart,
        pipeline_mode,
        parameter_overrides,
        snapshot_name,
        use_historical_intelligence,
        variation_config,
        start_date,
        end_date,
        submitted_by
      ) VALUES (
        CONCAT('tui_', TO_CHAR(NOW(), 'YYYYMMDD_HH24MISS_MS'), '_', SUBSTRING(MD5(RANDOM()::text) FROM 1 FOR 8)),
        'pending',
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13::jsonb, $14, $15, $16
      )
      RETURNING *
      `,
      [
        parsed.epic,
        parsed.days,
        parsed.strategy,
        parsed.timeframe,
        parsed.parallel,
        parsed.workers,
        parsed.chunk_days,
        parsed.generate_chart,
        parsed.pipeline_mode,
        JSON.stringify(parsed.parameter_overrides),
        parsed.snapshot_name,
        parsed.use_historical_intelligence,
        parsed.start_date,
        parsed.end_date,
        parsed.submitted_by,
        parsed.variation_config ? JSON.stringify(parsed.variation_config) : null,
      ]
    );

    return NextResponse.json({ job: insertResult.rows[0] }, { status: 201 });
  } catch (error) {
    console.error("Failed to queue forex backtest", error);
    return NextResponse.json({ error: "Failed to queue forex backtest" }, { status: 500 });
  }
}
