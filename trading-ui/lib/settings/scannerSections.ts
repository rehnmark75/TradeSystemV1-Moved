/**
 * Scanner settings section hierarchy.
 * Defines categories → subsections → fields for the unified scanner settings page.
 */

export interface ScannerSubsection {
  key: string;
  title: string;
  description?: string;
  fields: string[];
  critical?: boolean; // shows warning icon
}

export interface ScannerSection {
  key: string;
  label: string;
  icon: string;
}

export const SCANNER_ICONS: Record<string, string> = {
  core: "◈",
  indicators: "≋",
  "data-quality": "⊞",
  "trading-control": "⚡",
  "duplicate-detection": "⊟",
  "risk-management": "⚖",
  "trading-hours": "⏱",
  "order-executor": "◎",
  "smc-conflict": "◉",
  "claude-ai": "◐",
};

export const SCANNER_LABELS: Record<string, string> = {
  core: "Core Settings",
  indicators: "Indicators",
  "data-quality": "Data Quality",
  "trading-control": "Trading Control",
  "duplicate-detection": "Duplicate Detection",
  "risk-management": "Risk Management",
  "trading-hours": "Trading Hours",
  "order-executor": "Order Executor",
  "smc-conflict": "SMC Conflict",
  "claude-ai": "Claude AI",
};

export const SCANNER_SECTION_ORDER = [
  "core",
  "indicators",
  "data-quality",
  "trading-control",
  "duplicate-detection",
  "risk-management",
  "trading-hours",
  "order-executor",
  "smc-conflict",
  "claude-ai",
];

export const SCANNER_SUBSECTIONS: Record<string, ScannerSubsection[]> = {
  core: [
    {
      key: "scan-basics",
      title: "Scan Basics",
      fields: ["scan_interval", "scan_align_to_boundaries", "scan_boundary_offset_seconds"],
    },
    {
      key: "confidence",
      title: "Confidence",
      fields: ["min_confidence", "min_confluence_score"],
    },
    {
      key: "timeframes",
      title: "Timeframes",
      fields: ["default_timeframe"],
    },
    {
      key: "synthesis",
      title: "Multi-Timeframe Analysis",
      fields: ["use_1m_base_synthesis", "enable_multi_timeframe_analysis"],
    },
  ],

  indicators: [
    {
      key: "kama",
      title: "KAMA (Kaufman Adaptive Moving Average)",
      fields: ["kama_period", "kama_fast", "kama_slow"],
    },
    {
      key: "macd",
      title: "MACD",
      fields: ["macd_fast_period", "macd_slow_period", "macd_signal_period"],
    },
    {
      key: "bollinger",
      title: "Bollinger Bands",
      fields: ["bb_period", "bb_std_dev"],
    },
    {
      key: "supertrend",
      title: "Supertrend",
      fields: ["supertrend_period", "supertrend_multiplier"],
    },
    {
      key: "zero-lag",
      title: "Zero Lag EMA",
      fields: ["zero_lag_length", "zero_lag_band_mult"],
    },
    {
      key: "two-pole",
      title: "Two-Pole Filter",
      fields: ["two_pole_filter_length", "two_pole_sma_length", "two_pole_signal_delay"],
    },
  ],

  "data-quality": [
    {
      key: "fetching",
      title: "Data Fetching",
      description: "Controls how data is fetched. Incorrect values can cause insufficient data errors.",
      fields: ["enable_signal_freshness_check", "max_signal_age_minutes"],
      critical: true,
    },
    {
      key: "lookback",
      title: "Lookback Configuration",
      description: "Tune the lookback window and cache behavior for validation.",
      fields: [
        "lookback_reduction_factor",
        "min_bars_for_sr_analysis",
        "sr_lookback_hours",
        "sr_cache_duration_minutes",
      ],
    },
    {
      key: "quality",
      title: "Data Quality Filtering",
      description: "Gate trades when data quality is insufficient.",
      fields: [
        "enable_data_quality_filtering",
        "block_trading_on_data_issues",
        "min_quality_score_for_trading",
      ],
    },
    {
      key: "sr",
      title: "Support/Resistance Validation",
      description: "Controls structural validation for SR levels.",
      fields: [
        "enable_sr_validation",
        "enable_enhanced_sr_validation",
        "sr_analysis_timeframe",
        "sr_left_bars",
        "sr_right_bars",
        "sr_volume_threshold",
        "sr_level_tolerance_pips",
        "sr_min_level_distance_pips",
        "sr_recent_flip_bars",
        "sr_min_flip_strength",
      ],
    },
    {
      key: "news",
      title: "News + Regime Filters",
      description: "Reduce exposure around news and unsuitable regimes.",
      fields: [
        "enable_news_filtering",
        "reduce_confidence_near_news",
        "news_filter_fail_secure",
        "enable_market_intelligence_capture",
        "enable_market_intelligence_filtering",
        "market_intelligence_min_confidence",
        "market_intelligence_block_unsuitable_regimes",
        "market_bias_filter_enabled",
        "market_bias_min_consensus",
      ],
    },
  ],

  "trading-control": [
    {
      key: "master",
      title: "Master Controls",
      description: "Critical controls for live execution.",
      fields: ["auto_trading_enabled", "enable_order_execution"],
      critical: true,
    },
  ],

  "duplicate-detection": [
    {
      key: "presets",
      title: "Deduplication Preset",
      description: "Pre-configured settings for duplicate detection.",
      fields: ["deduplication_preset", "duplicate_sensitivity"],
    },
    {
      key: "toggles",
      title: "Detection Toggles",
      description: "Master toggles and baseline controls.",
      fields: [
        "enable_duplicate_check",
        "enable_alert_deduplication",
        "enable_price_similarity_check",
        "enable_strategy_cooldowns",
        "enable_signal_hash_check",
        "enable_time_based_hash_components",
      ],
    },
    {
      key: "cooldowns",
      title: "Cooldown Settings",
      description: "Control cooldowns per signal and strategy.",
      fields: [
        "signal_cooldown_minutes",
        "alert_cooldown_minutes",
        "strategy_cooldown_minutes",
        "global_cooldown_seconds",
        "deduplication_lookback_hours",
      ],
    },
    {
      key: "rate-limits",
      title: "Rate Limits",
      fields: ["max_alerts_per_hour", "max_alerts_per_epic_hour"],
    },
    {
      key: "similarity",
      title: "Similarity Thresholds",
      description: "Tuning thresholds for duplicate detection.",
      fields: ["price_similarity_threshold", "confidence_similarity_threshold"],
    },
    {
      key: "db-checks",
      title: "Database Checks",
      fields: [
        "use_database_dedup_check",
        "database_dedup_window_minutes",
        "deduplication_debug_mode",
      ],
    },
    {
      key: "cache",
      title: "Signal Hash Cache",
      description: "In-memory caching to reduce duplicate alerts.",
      fields: ["signal_hash_cache_expiry_minutes", "max_signal_hash_cache_size"],
    },
  ],

  "risk-management": [
    {
      key: "sizing",
      title: "Position Sizing",
      description: "Position size and account risk controls.",
      fields: [
        "position_size_percent",
        "min_position_size",
        "max_position_size",
        "risk_per_trade_percent",
      ],
    },
    {
      key: "sl-tp",
      title: "SL/TP and Limits",
      description: "Default stops, targets, and trade caps.",
      fields: [
        "stop_loss_pips",
        "take_profit_pips",
        "default_risk_reward",
        "max_open_positions",
        "max_daily_trades",
        "max_risk_per_trade",
      ],
    },
    {
      key: "guards",
      title: "Execution Guards",
      description: "Additional safety and spread constraints.",
      fields: [
        "default_stop_distance",
        "validate_spread",
        "max_spread_pips",
        "min_signal_confirmations",
        "scalping_min_confidence",
        "strategy_testing_mode",
      ],
    },
    {
      key: "pairs",
      title: "Pair Allow/Deny Lists",
      description: "Restrict trading to specific instruments.",
      fields: ["allowed_trading_epics", "blocked_trading_epics"],
    },
  ],

  "trading-hours": [
    {
      key: "schedule",
      title: "Trading Schedule",
      description: "Local trading hours and cutoff controls.",
      fields: [
        "trading_start_hour",
        "trading_end_hour",
        "trading_cutoff_time_utc",
        "user_timezone",
        "respect_market_hours",
        "weekend_scanning",
        "enable_trading_time_controls",
        "respect_trading_hours",
        "trade_cooldown_enabled",
        "trade_cooldown_minutes",
      ],
    },
    {
      key: "asian",
      title: "Asian Session (UTC)",
      fields: [
        "session_asian_start_hour",
        "session_asian_end_hour",
        "block_asian_session",
      ],
    },
    {
      key: "london",
      title: "London Session (UTC)",
      fields: ["session_london_start_hour", "session_london_end_hour"],
    },
    {
      key: "newyork",
      title: "New York Session (UTC)",
      fields: ["session_newyork_start_hour", "session_newyork_end_hour"],
    },
    {
      key: "overlap",
      title: "London/NY Overlap",
      description: "Best trading window overlap hours.",
      fields: ["session_overlap_start_hour", "session_overlap_end_hour"],
    },
  ],

  "order-executor": [
    {
      key: "confidence",
      title: "Confidence Thresholds",
      description: "Adjust stop sizing based on confidence.",
      fields: [
        "executor_high_confidence_threshold",
        "executor_medium_confidence_threshold",
      ],
    },
    {
      key: "multipliers",
      title: "Stop Distance Multipliers",
      description: "Tighten or widen stops by confidence level.",
      fields: [
        "executor_high_conf_stop_multiplier",
        "executor_low_conf_stop_multiplier",
      ],
    },
    {
      key: "limits",
      title: "SL/TP Sanity Limits",
      description: "Reject or cap orders outside these bounds.",
      fields: ["executor_max_stop_loss_pips", "executor_max_take_profit_pips"],
    },
  ],

  "smc-conflict": [
    {
      key: "analysis",
      title: "Smart Money Analysis",
      description: "Controls for SMC analysis runtime.",
      fields: ["smart_money_readonly_enabled", "smart_money_analysis_timeout"],
    },
    {
      key: "filter",
      title: "Conflict Filter",
      description: "Reject signals when SMC data conflicts.",
      fields: [
        "smc_conflict_filter_enabled",
        "smc_reject_order_flow_conflict",
        "smc_reject_ranging_structure",
      ],
    },
    {
      key: "thresholds",
      title: "Consensus Thresholds",
      description: "Minimum SMC consensus requirements.",
      fields: ["smc_min_directional_consensus", "smc_min_structure_score"],
    },
  ],

  "claude-ai": [
    {
      key: "master",
      title: "Master Controls",
      description: "Claude approvals and safety settings.",
      fields: [
        "require_claude_approval",
        "claude_fail_secure",
        "save_claude_rejections",
        "claude_validate_in_backtest",
      ],
    },
    {
      key: "model",
      title: "Model & Quality",
      description: "Choose model and minimum quality score.",
      fields: ["claude_model", "min_claude_quality_score"],
    },
    {
      key: "vision",
      title: "Vision Analysis",
      description: "Chart-based Claude analysis settings.",
      fields: [
        "claude_include_chart",
        "claude_vision_enabled",
        "claude_save_vision_artifacts",
        "claude_chart_timeframes",
        "claude_vision_strategies",
      ],
    },
  ],
};
