"use client";

import type { ChangeEvent } from "react";

interface SettingsFieldProps {
  label: string;
  name: string;
  value: unknown;
  defaultValue?: unknown;
  override?: boolean;
  pending?: boolean;
  unit?: string;
  description?: string;
  onChange: (value: unknown) => void;
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function parseInputValue(value: string, original: unknown) {
  if (typeof original === "number") {
    const next = Number(value);
    return Number.isNaN(next) ? value : next;
  }
  if (typeof original === "boolean") {
    return value === "true";
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

export default function SettingsField({
  label,
  name,
  value,
  defaultValue,
  override,
  pending,
  unit,
  description,
  onChange
}: SettingsFieldProps) {
  const valueType = typeof value;
  const isBoolean = valueType === "boolean";
  const isNumber = valueType === "number";
  const isObject = valueType === "object" && value !== null;

  const handleTextChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(parseInputValue(event.target.value, value));
  };

  const handleTextareaChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(parseInputValue(event.target.value, value));
  };

  return (
    <div className={`settings-field ${pending ? "is-pending" : ""}`}>
      <div className="settings-field-meta">
        <div>
          <div className="settings-field-label">
            {override ? <span className="override-indicator" /> : null}
            {label}
          </div>
          {description ? (
            <div className="settings-field-description">{description}</div>
          ) : null}
        </div>
        <div className="settings-field-default">
          <span>Default:</span>
          <strong>{formatValue(defaultValue)}</strong>
        </div>
      </div>
      <div className="settings-field-input">
        {isBoolean ? (
          <select
            aria-label={name}
            value={String(value)}
            onChange={(event) => onChange(event.target.value === "true")}
          >
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        ) : isObject ? (
          <textarea
            aria-label={name}
            value={formatValue(value)}
            onChange={handleTextareaChange}
            rows={4}
          />
        ) : (
          <input
            aria-label={name}
            type={isNumber ? "number" : "text"}
            value={formatValue(value)}
            onChange={handleTextChange}
          />
        )}
        {unit ? <span className="settings-field-unit">{unit}</span> : null}
      </div>
    </div>
  );
}
