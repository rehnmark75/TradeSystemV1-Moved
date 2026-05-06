export type JobRow = {
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

export type ExecutionRow = {
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

export type BacktestsPayload = {
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

export type LaunchFormState = {
  epic: string;
  days: number;
  strategy: string;
  timeframe: string;
  parallel: boolean;
  workers: number;
  chunk_days: number;
  generate_chart: boolean;
  pipeline_mode: boolean;
  use_historical_intelligence: boolean;
  start_date: string;
  end_date: string;
  snapshot_name: string;
  variation_enabled: boolean;
  variation_json: string;
  variation_workers: number;
  variation_rank_by: string;
  variation_top_n: number;
};
