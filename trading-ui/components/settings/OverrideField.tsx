"use client";

import type { ChangeEvent } from "react";

interface OverrideFieldProps {
  label: string;
  name: string;
  globalValue: unknown;
  effectiveValue: unknown;
  overrideValue: unknown;
  isOverridden: boolean;
  description?: string;
  unit?: string;
  dataType?: string;
  onToggle: (next: boolean) => void;
  onChange: (value: unknown) => void;
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

function parseInputValue(value: string, dataType?: string, original?: unknown) {
  const normalized = dataType?.toLowerCase() ?? "";
  if (normalized.includes("bool")) {
    return value === "true";
  }
  if (normalized.includes("int")) {
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? value : parsed;
  }
  if (normalized.includes("dec") || normalized.includes("num")) {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? value : parsed;
  }
  if (normalized.includes("json")) {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  if (typeof original === "object") {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
}

export default function OverrideField({
  label,
  name,
  globalValue,
  effectiveValue,
  overrideValue,
  isOverridden,
  description,
  unit,
  dataType,
  onToggle,
  onChange
}: OverrideFieldProps) {
  const normalizedType = dataType?.toLowerCase() ?? typeof effectiveValue;
  const isBoolean = normalizedType.includes("bool");
  const isNumber =
    normalizedType.includes("int") ||
    normalizedType.includes("dec") ||
    normalizedType.includes("num") ||
    typeof effectiveValue === "number";
  const isObject =
    normalizedType.includes("json") ||
    (typeof effectiveValue === "object" && effectiveValue !== null);

  const inputValue = isOverridden ? overrideValue : globalValue;
  const isOverrideActive = isOverridden;

  const displayValue = (() => {
    if (inputValue === null || inputValue === undefined) return "";
    if (typeof inputValue === "number") return inputValue;
    if (typeof inputValue === "boolean") return inputValue ? "true" : "false";
    if (typeof inputValue === "object") return JSON.stringify(inputValue);
    return String(inputValue);
  })();

  const handleTextChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(parseInputValue(event.target.value, dataType, inputValue));
  };

  const handleTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(parseInputValue(event.target.value, dataType, inputValue));
  };

  return (
    <div className={`override-field ${isOverrideActive ? "is-overridden" : ""}`}>
      <div className="override-field-meta">
        <div>
          <div className="override-field-label">{label}</div>
          {description ? (
            <div className="override-field-description">{description}</div>
          ) : null}
        </div>
        <div className="override-field-global">
          <span>Global</span>
          <strong>{formatValue(globalValue)}</strong>
        </div>
      </div>
      <div className="override-field-input">
        <button
          className={`override-toggle ${isOverrideActive ? "active" : ""}`}
          onClick={() => onToggle(!isOverrideActive)}
        >
          {isOverrideActive ? "Override" : "Inherit"}
        </button>
        {isBoolean ? (
          <select
            aria-label={name}
            value={String(inputValue ?? "false")}
            disabled={!isOverrideActive}
            onChange={(event) => onChange(event.target.value === "true")}
          >
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        ) : isObject ? (
          <textarea
            aria-label={name}
            value={isOverrideActive ? formatValue(inputValue) : formatValue(globalValue)}
            onChange={handleTextareaChange}
            rows={3}
            disabled={!isOverrideActive}
          />
        ) : (
          <input
            aria-label={name}
            type={isNumber ? "number" : "text"}
            value={displayValue}
            onChange={handleTextChange}
            disabled={!isOverrideActive}
          />
        )}
        {unit ? <span className="override-field-unit">{unit}</span> : null}
      </div>
      <div className="override-field-effective">
        Effective: <strong>{formatValue(effectiveValue)}</strong>
        <span className="override-field-tag">
          {isOverrideActive ? "Override" : "Inherited"}
        </span>
      </div>
    </div>
  );
}
