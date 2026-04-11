"use client";

import type { ChangeEvent } from "react";
import RangeSlider from "./RangeSlider";
import ToggleSwitch from "./ToggleSwitch";
import { getParamRiskLevel, getRiskColor } from "../../lib/settings/riskClassification";
import type { SmcParameterMetadata } from "../../types/settings";

interface SettingsFieldProps {
  name: string;
  label: string;
  value: unknown;
  defaultValue?: unknown;
  description?: string;
  unit?: string;
  pending?: boolean;
  override?: boolean;
  metadata?: SmcParameterMetadata;
  // Override mode: show global/effective and toggle
  overrideMode?: boolean;
  globalValue?: unknown;
  effectiveValue?: unknown;
  isOverridden?: boolean;
  onToggle?: (next: boolean) => void;
  onFocus?: () => void;
  onChange: (value: unknown) => void;
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function formatDisplayValue(value: unknown) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function parseInputValue(value: string, original: unknown, dataType?: string) {
  const dt = dataType?.toLowerCase() ?? "";
  if (dt.includes("bool") || typeof original === "boolean") {
    return value === "true";
  }
  if (dt.includes("int")) {
    const parsed = parseInt(value, 10);
    return isNaN(parsed) ? value : parsed;
  }
  if (dt.includes("dec") || dt.includes("num") || typeof original === "number") {
    const parsed = Number(value);
    return isNaN(parsed) ? value : parsed;
  }
  if (dt.includes("json") || (typeof original === "object" && original !== null)) {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
}

function getDeviationLevel(value: unknown, defaultValue: unknown): "none" | "low" | "high" {
  if (defaultValue === undefined || defaultValue === null) return "none";
  if (typeof value !== "number" || typeof defaultValue !== "number") return "none";
  const pct = Math.abs((value - defaultValue) / (defaultValue || 1)) * 100;
  if (pct > 50) return "high";
  if (pct > 15) return "low";
  return "none";
}

export default function SettingsField({
  name,
  label,
  value,
  defaultValue,
  description,
  unit,
  pending,
  override,
  metadata,
  overrideMode = false,
  globalValue,
  effectiveValue,
  isOverridden = false,
  onToggle,
  onFocus,
  onChange,
}: SettingsFieldProps) {
  const dataType = metadata?.data_type ?? "";
  const minValue = metadata?.min_value !== undefined && metadata?.min_value !== null
    ? Number(metadata.min_value)
    : undefined;
  const maxValue = metadata?.max_value !== undefined && metadata?.max_value !== null
    ? Number(metadata.max_value)
    : undefined;
  const validOptions = metadata?.valid_options as string[] | null | undefined;
  const requiresRestart = metadata?.requires_restart ?? false;

  const riskLevel = getParamRiskLevel(name, requiresRestart ?? false);
  const riskColor = getRiskColor(riskLevel);

  // In override mode, use the override value if overridden, else fall back to global
  const editValue = overrideMode && !isOverridden ? globalValue : value;

  const valueType = dataType.toLowerCase();
  const isBoolean =
    valueType.includes("bool") ||
    typeof editValue === "boolean";
  const isNumber =
    valueType.includes("int") ||
    valueType.includes("dec") ||
    valueType.includes("num") ||
    typeof editValue === "number";
  const isObject =
    valueType.includes("json") ||
    (typeof editValue === "object" && editValue !== null);

  const hasRange = isNumber && minValue !== undefined && maxValue !== undefined;
  const hasOptions = Array.isArray(validOptions) && validOptions.length > 0;

  const deviation = getDeviationLevel(
    overrideMode ? value : editValue,
    defaultValue
  );

  const defaultNum =
    defaultValue !== undefined && defaultValue !== null
      ? Number(defaultValue)
      : undefined;

  const handleTextChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(parseInputValue(event.target.value, editValue, dataType));
  };

  const handleTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(parseInputValue(event.target.value, editValue, dataType));
  };

  const inputDisabled = overrideMode && !isOverridden;

  return (
    <div
      className={[
        "settings-field",
        pending ? "is-pending" : "",
        overrideMode && isOverridden ? "is-overridden" : "",
        `risk-${riskLevel}`,
        `deviation-${deviation}`,
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ "--risk-color": riskColor } as React.CSSProperties}
      onFocus={onFocus}
      tabIndex={-1}
    >
      <div className="settings-field-meta">
        <div className="settings-field-meta-left">
          <div className="settings-field-label">
            {override ? <span className="override-indicator" /> : null}
            {riskLevel !== "normal" ? (
              <span
                className={`risk-badge risk-badge--${riskLevel}`}
                title={riskLevel === "critical" ? "Critical setting" : "High impact"}
              />
            ) : null}
            {label}
            {requiresRestart ? (
              <span className="restart-badge" title="Requires restart">↺</span>
            ) : null}
          </div>
          {description ? (
            <div className="settings-field-description">{description}</div>
          ) : null}
        </div>
        <div className="settings-field-meta-right">
          {defaultValue !== undefined ? (
            <div className="settings-field-default">
              <span>Default</span>
              <strong>{formatDisplayValue(defaultValue)}</strong>
            </div>
          ) : null}
          {overrideMode && globalValue !== undefined ? (
            <div className="settings-field-global">
              <span>Global</span>
              <strong>{formatDisplayValue(globalValue)}</strong>
            </div>
          ) : null}
        </div>
      </div>

      <div className="settings-field-input">
        {overrideMode && onToggle ? (
          <button
            type="button"
            className={`override-toggle ${isOverridden ? "active" : ""}`}
            onClick={() => onToggle(!isOverridden)}
          >
            {isOverridden ? "Override" : "Inherit"}
          </button>
        ) : null}

        {isBoolean ? (
          <ToggleSwitch
            checked={Boolean(editValue)}
            onChange={(checked) => onChange(checked)}
            disabled={inputDisabled}
          />
        ) : hasOptions ? (
          <select
            aria-label={name}
            value={formatValue(editValue)}
            disabled={inputDisabled}
            onChange={(e) => onChange(e.target.value)}
          >
            {validOptions!.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        ) : isObject ? (
          <textarea
            aria-label={name}
            value={formatValue(editValue)}
            onChange={handleTextareaChange}
            disabled={inputDisabled}
            rows={3}
          />
        ) : hasRange ? (
          <RangeSlider
            value={typeof editValue === "number" ? editValue : Number(editValue) || 0}
            min={minValue!}
            max={maxValue!}
            defaultValue={defaultNum}
            unit={unit}
            disabled={inputDisabled}
            onChange={(v) => onChange(v)}
          />
        ) : (
          <>
            <input
              aria-label={name}
              type={isNumber ? "number" : "text"}
              value={formatValue(editValue)}
              onChange={handleTextChange}
              disabled={inputDisabled}
            />
            {unit && !hasRange ? (
              <span className="settings-field-unit">{unit}</span>
            ) : null}
          </>
        )}
      </div>

      {overrideMode ? (
        <div className="settings-field-effective">
          Effective: <strong>{formatDisplayValue(effectiveValue ?? globalValue)}</strong>
          <span className="override-field-tag">
            {isOverridden ? "Override" : "Inherited"}
          </span>
        </div>
      ) : null}
    </div>
  );
}
