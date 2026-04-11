"use client";

import { useId } from "react";

interface RangeSliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  defaultValue?: number;
  unit?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}

export default function RangeSlider({
  value,
  min,
  max,
  step,
  defaultValue,
  unit,
  disabled = false,
  onChange,
}: RangeSliderProps) {
  const id = useId();

  // Compute step: if not provided, derive a sensible default
  const effectiveStep = step ?? deriveStep(min, max);

  // Percentage position of current value and default value on track
  const pct = clamp(((value - min) / (max - min)) * 100, 0, 100);
  const defaultPct =
    defaultValue !== undefined
      ? clamp(((defaultValue - min) / (max - min)) * 100, 0, 100)
      : null;

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value));
  };

  const handleNumber = (e: React.ChangeEvent<HTMLInputElement>) => {
    const parsed = Number(e.target.value);
    if (!Number.isNaN(parsed)) {
      onChange(clamp(parsed, min, max));
    }
  };

  return (
    <div className="range-slider">
      <div className="range-slider-track-wrap">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={effectiveStep}
          value={value}
          disabled={disabled}
          onChange={handleSlider}
          className="range-slider-input"
          style={{ "--fill-pct": `${pct}%` } as React.CSSProperties}
        />
        {defaultPct !== null ? (
          <span
            className="range-slider-default-tick"
            style={{ left: `${defaultPct}%` }}
            title={`Default: ${defaultValue}`}
          />
        ) : null}
      </div>
      <div className="range-slider-companion">
        <input
          type="number"
          min={min}
          max={max}
          step={effectiveStep}
          value={value}
          disabled={disabled}
          onChange={handleNumber}
          className="range-slider-number"
        />
        {unit ? <span className="range-slider-unit">{unit}</span> : null}
      </div>
    </div>
  );
}

function clamp(val: number, min: number, max: number) {
  return Math.min(Math.max(val, min), max);
}

function deriveStep(min: number, max: number): number {
  const range = max - min;
  if (range <= 1) return 0.01;
  if (range <= 10) return 0.1;
  if (range <= 100) return 1;
  return Math.round(range / 100);
}
