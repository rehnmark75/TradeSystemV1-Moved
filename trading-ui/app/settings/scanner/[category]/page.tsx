"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import SettingsField from "../../../../components/settings/SettingsField";
import SettingsForm from "../../../../components/settings/SettingsForm";
import SettingsGroup from "../../../../components/settings/SettingsGroup";
import SettingsSearch from "../../../../components/settings/SettingsSearch";
import ConflictModal from "../../../../components/settings/ConflictModal";
import { scannerCategories } from "../../../../lib/settings/scannerCategories";
import { useScannerConfig } from "../../../../hooks/settings/useScannerConfig";
import { useSettingsSearch } from "../../../../hooks/settings/useSettingsSearch";
import { logTelemetry } from "../../../../lib/settings/telemetry";

interface ScannerCategoryPageProps {
  params: { category: string };
}

function toLabel(value: string) {
  return value.replace(/_/g, " ");
}

function coreGroupForField(field: string) {
  if (field.startsWith("scan_") || field.includes("scan_")) return "scan";
  if (field.includes("confidence") || field.includes("confluence")) return "confidence";
  if (field.includes("timeframe")) return "timeframe";
  if (field.includes("synthesis") || field.includes("multi_timeframe")) return "synthesis";
  if (field.includes("cooldown") || field.includes("dedup")) return "control";
  return "other";
}

const coreGroupLabels: Record<string, string> = {
  scan: "Scan Basics",
  confidence: "Confidence",
  timeframe: "Timeframes",
  synthesis: "Multi-Timeframe Analysis",
  control: "Controls",
  other: "Other"
};

const scannerTabs = [
  { key: "core", label: "Core Settings" },
  { key: "indicators", label: "Indicators" },
  { key: "data-quality", label: "Data Quality" },
  { key: "trading-control", label: "Trading Control" },
  { key: "duplicate-detection", label: "Duplicate Detection" },
  { key: "risk-management", label: "Risk Management" },
  { key: "trading-hours", label: "Trading Hours" },
  { key: "order-executor", label: "Order Executor" },
  { key: "smc-conflict", label: "SMC Conflict" },
  { key: "claude-ai", label: "Claude AI" }
];

const indicatorSectionOrder = [
  "kama",
  "macd",
  "bb-supertrend",
  "zero-lag",
  "two-pole",
  "other"
];

const indicatorSectionLabels: Record<string, string> = {
  kama: "KAMA (Kaufman Adaptive Moving Average)",
  macd: "MACD",
  "bb-supertrend": "Bollinger Bands & Supertrend",
  "zero-lag": "Zero Lag EMA",
  "two-pole": "Two-Pole Filter",
  other: "Other Indicators"
};

const indicatorSubgroups: Record<string, { label: string; fields: string[] }[]> = {
  "bb-supertrend": [
    {
      label: "Bollinger Bands",
      fields: ["bb_period", "bb_std_dev"]
    },
    {
      label: "Supertrend",
      fields: ["supertrend_period", "supertrend_multiplier"]
    }
  ]
};

const dataQualitySections = [
  {
    key: "fetching",
    title: "Data Fetching Settings (Critical)",
    description:
      "These settings control how much data is fetched. Incorrect values can cause insufficient data errors.",
    fields: ["enable_signal_freshness_check", "max_signal_age_minutes"]
  },
  {
    key: "lookback",
    title: "Lookback Configuration",
    description: "Tune the lookback window and cache behavior for validation.",
    fields: ["lookback_reduction_factor", "min_bars_for_sr_analysis", "sr_lookback_hours", "sr_cache_duration_minutes"]
  },
  {
    key: "quality",
    title: "Data Quality Filtering",
    description: "Gate trades when data quality is insufficient.",
    fields: ["enable_data_quality_filtering", "block_trading_on_data_issues", "min_quality_score_for_trading"]
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
      "sr_min_flip_strength"
    ]
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
      "market_bias_min_consensus"
    ]
  }
];

export default function ScannerCategoryPage({ params }: ScannerCategoryPageProps) {
  const [query, setQuery] = useState("");
  const [modifiedOnly, setModifiedOnly] = useState(false);
  const [updatedBy, setUpdatedBy] = useState("");
  const [changeReason, setChangeReason] = useState("");
  const {
    effectiveData,
    defaults,
    loading,
    error,
    changes,
    updateField,
    saveChanges,
    resetChanges,
    conflict,
    setConflict,
    setData,
    setChanges
  } = useScannerConfig();

  const categoryKey = params.category;
  const fields = scannerCategories[categoryKey] ?? [];
  const dataKeys = effectiveData ? Object.keys(effectiveData as Record<string, unknown>) : [];
  const knownFields = Object.values(scannerCategories).flat();
  const unassignedFields = dataKeys.filter(
    (key) => !knownFields.includes(key) && !["id", "version", "created_at"].includes(key)
  );
  const finalFields = fields;
  const data = effectiveData as Record<string, unknown> | null;
  const filtered = useSettingsSearch(finalFields, data, defaults, query, modifiedOnly);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: `/settings/scanner/${categoryKey}` });
  }, [categoryKey]);

  const isCore = categoryKey === "core";
  const isIndicators = categoryKey === "indicators";
  const isDataQuality = categoryKey === "data-quality";
  const isTradingControl = categoryKey === "trading-control";
  const isDuplicate = categoryKey === "duplicate-detection";
  const isRisk = categoryKey === "risk-management";
  const isTradingHours = categoryKey === "trading-hours";
  const isOrderExecutor = categoryKey === "order-executor";
  const isSmcConflict = categoryKey === "smc-conflict";
  const isClaude = categoryKey === "claude-ai";
  const useStackLayout =
    isIndicators ||
    isDataQuality ||
    isTradingControl ||
    isDuplicate ||
    isRisk ||
    isTradingHours ||
    isOrderExecutor ||
    isSmcConflict ||
    isClaude;
  const categoryLabel = toLabel(categoryKey);
  const version = String(data?.version ?? "-");
  const status = data?.is_active ? "Active" : "Inactive";
  const updatedAt = data?.updated_at
    ? new Date(String(data.updated_at)).toLocaleString()
    : "-";
  const scanInterval = data?.scan_interval ? `${data.scan_interval}s` : "-";

  const content = useMemo(() => {
    if (!data) return null;
    const renderField = (field: string) => (
      <SettingsField
        key={field}
        name={field}
        label={toLabel(field)}
        value={data[field]}
        defaultValue={defaults[field]}
        override={defaults[field] !== undefined && data[field] !== defaults[field]}
        pending={field in changes}
        onChange={(value) => updateField(field, value)}
      />
    );

    const renderSection = (
      key: string,
      title: string,
      description: string | null,
      fields: string[],
      gridClass: string = "grid-2"
    ) => {
      const visible = fields.filter((field) => filtered.includes(field));
      if (!visible.length) return null;
      return (
        <div key={key} className="scanner-section-block">
          <div className="scanner-section-header">
            <h3>{title}</h3>
            {description ? <p>{description}</p> : null}
          </div>
          <div className={`scanner-section-grid ${gridClass}`}>
            {visible.map(renderField)}
          </div>
        </div>
      );
    };

    const renderSubSections = (
      key: string,
      title: string,
      description: string | null,
      sections: Array<{ label: string; fields: string[] }>,
      gridClass: string = "grid-2"
    ) => {
      const filteredSections = sections
        .map((section) => ({
          ...section,
          fields: section.fields.filter((field) => filtered.includes(field))
        }))
        .filter((section) => section.fields.length);
      if (!filteredSections.length) return null;
      return (
        <div key={key} className="scanner-section-block">
          <div className="scanner-section-header">
            <h3>{title}</h3>
            {description ? <p>{description}</p> : null}
          </div>
          <div className={`scanner-section-subgrid ${gridClass}`}>
            {filteredSections.map((section) => (
              <div key={section.label} className="scanner-subsection">
                <div className="scanner-subsection-title">{section.label}</div>
                <div className="scanner-section-grid grid-2">
                  {section.fields.map(renderField)}
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    };

    if (isIndicators) {
      const groups = new Map<string, string[]>();
      filtered.forEach((field) => {
        let group = "other";
        if (field.startsWith("kama_")) group = "kama";
        else if (field.startsWith("macd_")) group = "macd";
        else if (field.startsWith("bb_") || field.startsWith("supertrend_")) group = "bb-supertrend";
        else if (field.startsWith("zero_lag_")) group = "zero-lag";
        else if (field.startsWith("two_pole_")) group = "two-pole";
        const bucket = groups.get(group) ?? [];
        bucket.push(field);
        groups.set(group, bucket);
      });

      const nodes: JSX.Element[] = [];
      indicatorSectionOrder.forEach((sectionKey) => {
        const sectionFields = groups.get(sectionKey);
        if (!sectionFields?.length && !indicatorSubgroups[sectionKey]) return;

        nodes.push(
          <div key={`${sectionKey}-title`} className="indicator-section-title">
            {indicatorSectionLabels[sectionKey] ?? sectionKey}
          </div>
        );

        const subgroups = indicatorSubgroups[sectionKey];
        if (subgroups) {
          nodes.push(
            <div key={`${sectionKey}-subgroups`} className="indicator-subgrid">
              {subgroups.map((subgroup) => (
                <div key={subgroup.label} className="indicator-subcard">
                  <div className="indicator-subtitle">{subgroup.label}</div>
                  <div className="indicator-grid">
                    {subgroup.fields
                      .filter((field) => filtered.includes(field))
                      .map((field) => (
                        <SettingsField
                          key={field}
                          name={field}
                          label={toLabel(field)}
                          value={data[field]}
                          defaultValue={defaults[field]}
                          override={
                            defaults[field] !== undefined && data[field] !== defaults[field]
                          }
                          pending={field in changes}
                          onChange={(value) => updateField(field, value)}
                        />
                      ))}
                  </div>
                </div>
              ))}
            </div>
          );
          return;
        }

        nodes.push(
          <div key={`${sectionKey}-grid`} className="indicator-grid">
            {(sectionFields ?? []).map((field) => (
              <SettingsField
                key={field}
                name={field}
                label={toLabel(field)}
                value={data[field]}
                defaultValue={defaults[field]}
                override={defaults[field] !== undefined && data[field] !== defaults[field]}
                pending={field in changes}
                onChange={(value) => updateField(field, value)}
              />
            ))}
          </div>
        );
      });

      return nodes;
    }

    if (isDataQuality) {
      const nodes: JSX.Element[] = [];
      dataQualitySections.forEach((section) => {
        const visibleFields = section.fields.filter((field) => filtered.includes(field));
        if (!visibleFields.length) return;
        nodes.push(
          <div key={`${section.key}-header`} className="data-quality-header">
            <div className="data-quality-title">
              {section.key === "fetching" ? <span className="dq-alert">⚠</span> : null}
              <h3>{section.title}</h3>
            </div>
            <p>{section.description}</p>
          </div>
        );
        nodes.push(
          <div key={`${section.key}-grid`} className="data-quality-grid">
            {visibleFields.map(renderField)}
          </div>
        );
      });
      return nodes;
    }

    if (isTradingControl) {
      const autoEnabled = !!data.auto_trading_enabled;
      const orderEnabled = !!data.enable_order_execution;
      const liveTrading = autoEnabled || orderEnabled;
      return [
        <div
          key="trading-banner"
          className={`scanner-status-banner ${liveTrading ? "danger" : "safe"}`}
        >
          {liveTrading
            ? "Live trading is enabled — real orders will be executed."
            : "Live trading is disabled — signals only, no orders executed."}
        </div>,
        renderSection(
          "trading-master",
          "Master Trading Controls",
          "Critical controls for live execution.",
          ["auto_trading_enabled", "enable_order_execution"],
          "grid-2"
        ),
        <div key="trading-status" className="scanner-status-row">
          <span className={`status-pill ${autoEnabled ? "on" : "off"}`}>
            Auto Trading: {autoEnabled ? "Enabled" : "Disabled"}
          </span>
          <span className={`status-pill ${orderEnabled ? "on" : "off"}`}>
            Order Execution: {orderEnabled ? "Enabled" : "Disabled"}
          </span>
          <span className={`status-pill ${liveTrading ? "warn" : "off"}`}>
            Live Trading: {liveTrading ? "Active" : "Inactive"}
          </span>
        </div>
      ].filter(Boolean) as JSX.Element[];
    }

    if (isDuplicate) {
      return [
        renderSection(
          "dedup-presets",
          "Deduplication Preset",
          "Pre-configured settings for duplicate detection.",
          ["deduplication_preset", "duplicate_sensitivity"],
          "grid-2"
        ),
        renderSection(
          "dedup-toggle",
          "Duplicate Detection",
          "Master toggle and baseline controls.",
          ["enable_duplicate_check", "enable_alert_deduplication"],
          "grid-2"
        ),
        renderSubSections(
          "dedup-cooldowns",
          "Cooldown Settings",
          "Control cooldowns per signal and strategy.",
          [
            {
              label: "Signal Cooldowns",
              fields: [
                "signal_cooldown_minutes",
                "alert_cooldown_minutes",
                "strategy_cooldown_minutes",
                "global_cooldown_seconds"
              ]
            },
            {
              label: "Rate Limits",
              fields: ["max_alerts_per_hour", "max_alerts_per_epic_hour"]
            }
          ],
          "grid-2"
        ),
        renderSubSections(
          "dedup-advanced",
          "Advanced Deduplication",
          "Tuning thresholds and database checks.",
          [
            {
              label: "Similarity Thresholds",
              fields: ["price_similarity_threshold", "confidence_similarity_threshold"]
            },
            {
              label: "Database Checks",
              fields: ["use_database_dedup_check", "database_dedup_window_minutes"]
            },
            {
              label: "Debugging",
              fields: ["deduplication_debug_mode", "enable_signal_hash_check", "enable_time_based_hash_components"]
            }
          ],
          "grid-2"
        ),
        renderSection(
          "dedup-cache",
          "Signal Hash Cache",
          "In-memory caching to reduce duplicate alerts.",
          ["signal_hash_cache_expiry_minutes", "max_signal_hash_cache_size"],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (isRisk) {
      return [
        renderSection(
          "risk-sizing",
          "Position Sizing",
          "Position size and account risk controls.",
          ["position_size_percent", "min_position_size", "max_position_size", "risk_per_trade_percent"],
          "grid-2"
        ),
        renderSection(
          "risk-stops",
          "SL/TP and Limits",
          "Default stops, targets, and trade caps.",
          [
            "stop_loss_pips",
            "take_profit_pips",
            "default_risk_reward",
            "max_open_positions",
            "max_daily_trades",
            "max_risk_per_trade"
          ],
          "grid-2"
        ),
        renderSection(
          "risk-guards",
          "Execution Guards",
          "Additional safety and spread constraints.",
          [
            "default_stop_distance",
            "validate_spread",
            "max_spread_pips",
            "min_signal_confirmations",
            "scalping_min_confidence",
            "strategy_testing_mode"
          ],
          "grid-2"
        ),
        renderSection(
          "risk-epics",
          "Pair Allow/Deny Lists",
          "Restrict trading to specific instruments.",
          ["allowed_trading_epics", "blocked_trading_epics"],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (isTradingHours) {
      return [
        renderSubSections(
          "hours-basics",
          "Trading Hours",
          "Local trading hours and cutoff controls.",
          [
            {
              label: "Trading Hours",
              fields: ["trading_start_hour", "trading_end_hour", "trading_cutoff_time_utc", "user_timezone"]
            },
            {
              label: "Controls",
              fields: [
                "respect_market_hours",
                "weekend_scanning",
                "enable_trading_time_controls",
                "trade_cooldown_enabled",
                "trade_cooldown_minutes",
                "respect_trading_hours"
              ]
            }
          ],
          "grid-2"
        ),
        renderSubSections(
          "hours-sessions",
          "Forex Session Hours (UTC)",
          "Configure trading session boundaries.",
          [
            {
              label: "Asian Session",
              fields: ["session_asian_start_hour", "session_asian_end_hour", "block_asian_session"]
            },
            {
              label: "London Session",
              fields: ["session_london_start_hour", "session_london_end_hour"]
            },
            {
              label: "New York Session",
              fields: ["session_newyork_start_hour", "session_newyork_end_hour"]
            }
          ],
          "grid-3"
        ),
        renderSection(
          "hours-overlap",
          "London/NY Overlap",
          "Best trading window overlap hours.",
          ["session_overlap_start_hour", "session_overlap_end_hour"],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (isOrderExecutor) {
      return [
        renderSection(
          "executor-confidence",
          "Confidence Thresholds",
          "Adjust stop sizing based on confidence.",
          ["executor_high_confidence_threshold", "executor_medium_confidence_threshold"],
          "grid-2"
        ),
        renderSection(
          "executor-multipliers",
          "Stop Distance Multipliers",
          "Tighten or widen stops by confidence.",
          ["executor_high_conf_stop_multiplier", "executor_low_conf_stop_multiplier"],
          "grid-2"
        ),
        renderSection(
          "executor-limits",
          "SL/TP Sanity Limits",
          "Reject or cap orders outside these bounds.",
          ["executor_max_stop_loss_pips", "executor_max_take_profit_pips"],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (isSmcConflict) {
      return [
        renderSection(
          "smc-analysis",
          "Smart Money Analysis",
          "Controls for SMC analysis runtime.",
          ["smart_money_readonly_enabled", "smart_money_analysis_timeout"],
          "grid-2"
        ),
        renderSection(
          "smc-filter",
          "Conflict Filter",
          "Reject signals when SMC data conflicts.",
          [
            "smc_conflict_filter_enabled",
            "smc_reject_order_flow_conflict",
            "smc_reject_ranging_structure"
          ],
          "grid-2"
        ),
        renderSection(
          "smc-thresholds",
          "Consensus Thresholds",
          "Minimum SMC consensus requirements.",
          ["smc_min_directional_consensus", "smc_min_structure_score"],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (isClaude) {
      return [
        renderSection(
          "claude-master",
          "Master Controls",
          "Claude approvals and safety settings.",
          [
            "require_claude_approval",
            "claude_fail_secure",
            "save_claude_rejections",
            "claude_validate_in_backtest"
          ],
          "grid-2"
        ),
        renderSection(
          "claude-model",
          "Model & Quality",
          "Choose model and minimum score.",
          ["claude_model", "min_claude_quality_score"],
          "grid-2"
        ),
        renderSection(
          "claude-vision",
          "Vision Analysis",
          "Chart-based Claude analysis settings.",
          [
            "claude_include_chart",
            "claude_vision_enabled",
            "claude_save_vision_artifacts",
            "claude_chart_timeframes",
            "claude_vision_strategies"
          ],
          "grid-2"
        )
      ].filter(Boolean) as JSX.Element[];
    }

    if (!isCore) {
      return filtered.map(renderField);
    }

    const grouped = new Map<string, string[]>();
    const order = ["scan", "confidence", "timeframe", "synthesis", "control", "other"];
    filtered.forEach((field) => {
      const group = coreGroupForField(field);
      const bucket = grouped.get(group) ?? [];
      bucket.push(field);
      grouped.set(group, bucket);
    });

    const nodes: JSX.Element[] = [];
    order.forEach((groupKey) => {
      const fields = grouped.get(groupKey);
      if (!fields?.length) return;
      nodes.push(
        <div key={`${groupKey}-header`} className="scanner-section-header">
          {coreGroupLabels[groupKey] ?? groupKey}
        </div>
      );
      fields.forEach((field) => {
        nodes.push(
          <SettingsField
            key={field}
            name={field}
            label={toLabel(field)}
            value={data[field]}
            defaultValue={defaults[field]}
            override={defaults[field] !== undefined && data[field] !== defaults[field]}
            pending={field in changes}
            onChange={(value) => updateField(field, value)}
          />
        );
      });
    });

    return nodes;
  }, [
    data,
    filtered,
    defaults,
    changes,
    updateField,
    isCore,
    isIndicators,
    isDataQuality,
    isTradingControl,
    isDuplicate,
    isRisk,
    isTradingHours,
    isOrderExecutor,
    isSmcConflict,
    isClaude,
    categoryLabel
  ]);

  if (loading) {
    return <div className="settings-panel">Loading settings...</div>;
  }

  if (categoryKey === "audit") {
    return (
      <div className="settings-panel">
        <div className="settings-hero">
          <h1>Scanner Audit</h1>
          <p>View the full audit trail in the audit section.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`settings-panel scanner-category-${categoryKey}`}>
      <div className="settings-hero">
        <h1>Forex Scanner Settings</h1>
        <p>Database-driven configuration for the live scanner.</p>
      </div>
      <div className="scanner-metrics">
        <div className="scanner-metric">
          <span>Version</span>
          <strong>{version}</strong>
        </div>
        <div className="scanner-metric">
          <span>Status</span>
          <strong>{status}</strong>
        </div>
        <div className="scanner-metric">
          <span>Last Updated</span>
          <strong>{updatedAt}</strong>
        </div>
        <div className="scanner-metric">
          <span>Scan Interval</span>
          <strong>{scanInterval}</strong>
        </div>
      </div>
      <div className="scanner-tabs">
        {scannerTabs.map((tab) => (
          <Link
            key={tab.key}
            href={`/settings/scanner/${tab.key}`}
            className={`scanner-tab ${tab.key === categoryKey ? "active" : ""}`}
          >
            {tab.label}
          </Link>
        ))}
        <Link
          href="/settings/scanner/audit"
          className={`scanner-tab ${categoryKey === "audit" ? "active" : ""}`}
        >
          Audit
        </Link>
      </div>
      <div className="settings-hero">
        <h1>{categoryLabel} Settings</h1>
        <p>Category: {categoryLabel}</p>
      </div>
      {error ? <div className="settings-placeholder">Error: {error}</div> : null}
      <SettingsSearch value={query} onChange={setQuery} />
      <div className="settings-toggle">
        <label>
          <input
            type="checkbox"
            checked={modifiedOnly}
            onChange={(event) => setModifiedOnly(event.target.checked)}
          />
          Show modified only
        </label>
      </div>
      {isCore ? (
        <SettingsGroup title="Core Settings">
          <div className="settings-box-grid scanner-core-form">{content}</div>
        </SettingsGroup>
      ) : (
        <SettingsGroup title={`${categoryLabel} Settings`}>
          <div className={useStackLayout ? "scanner-stack" : "settings-box-grid scanner-core-form"}>
            {content}
          </div>
        </SettingsGroup>
      )}
      <SettingsForm
        title="Scanner Updates"
        changes={changes}
        updatedBy={updatedBy}
        changeReason={changeReason}
        onUpdatedByChange={setUpdatedBy}
        onChangeReasonChange={setChangeReason}
        onSave={({ updatedBy, changeReason }) =>
          saveChanges({ updatedBy, changeReason })
        }
        onRevert={resetChanges}
        onDiscard={resetChanges}
      />
      <ConflictModal
        open={!!conflict}
        current={conflict}
        pending={changes}
        onClose={() => setConflict(null)}
        onResolve={async ({ action, mergedChanges }) => {
          if (!conflict) return;
          if (action === "discard") {
            setData(conflict as any);
            resetChanges();
            setConflict(null);
            return;
          }
          if (mergedChanges) {
            if (!updatedBy.trim() || !changeReason.trim()) {
              alert("Updated by and change reason are required.");
              return;
            }
            setData(conflict as any);
            setChanges(mergedChanges);
            await saveChanges(
              { updatedBy, changeReason },
              mergedChanges,
              (conflict as any).updated_at
            );
            setConflict(null);
          }
        }}
      />
    </div>
  );
}
