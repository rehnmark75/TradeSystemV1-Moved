"use client";

import { useEffect, useRef } from "react";

export interface CategoryNavItem {
  category: string;
  fieldCount: number;
  modifiedCount: number;
  overriddenCount: number;
}

interface CategoryNavProps {
  items: CategoryNavItem[];
  activeCategory?: string;
  onSelect: (category: string) => void;
  showSnapshots?: boolean;
  onSnapshotsClick?: () => void;
  snapshotCount?: number;
}

const CATEGORY_ICONS: Record<string, string> = {
  "Tier 1: 4H Directional Bias": "◈",
  "Tier 2: 15m Entry Trigger": "◉",
  "Tier 3: 5m Execution": "◎",
  "Risk Management": "⚖",
  "Session Filter": "⏱",
  "Confidence Scoring": "◐",
  "MACD Alignment Filter": "≋",
  "Swing Proximity Validation (TIER 4)": "⊞",
  "Adaptive Cooldown": "↺",
  "Scalp Mode (High-Frequency Trading)": "⚡",
  "Scalp": "⚡",
  "Scalp Qualification": "✦",
  "Alternative Triggers": "⊕",
  "Enabled Trading Pairs": "⊟",
  Other: "⋯",
};

function getIcon(category: string): string {
  return CATEGORY_ICONS[category] ?? "◦";
}

function shortLabel(category: string): string {
  const replacements: Record<string, string> = {
    "Tier 1: 4H Directional Bias": "Tier 1 · 4H Bias",
    "Tier 2: 15m Entry Trigger": "Tier 2 · Entry",
    "Tier 3: 5m Execution": "Tier 3 · Execution",
    "Swing Proximity Validation (TIER 4)": "Swing Proximity",
    "Scalp Mode (High-Frequency Trading)": "Scalp Mode",
  };
  return replacements[category] ?? category;
}

export default function CategoryNav({
  items,
  activeCategory,
  onSelect,
  showSnapshots = false,
  onSnapshotsClick,
  snapshotCount = 0,
}: CategoryNavProps) {
  const navRef = useRef<HTMLDivElement>(null);

  // Scroll active item into view
  useEffect(() => {
    if (!activeCategory || !navRef.current) return;
    const active = navRef.current.querySelector(".category-nav-item.active");
    if (active) {
      active.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [activeCategory]);

  return (
    <div className="category-nav" ref={navRef}>
      <div className="category-nav-title">Categories</div>
      <div className="category-nav-list">
        {items.map((item) => {
          const isActive = item.category === activeCategory;
          return (
            <button
              key={item.category}
              type="button"
              className={`category-nav-item ${isActive ? "active" : ""}`}
              onClick={() => onSelect(item.category)}
              title={item.category}
            >
              <span className="category-nav-icon">{getIcon(item.category)}</span>
              <span className="category-nav-label">{shortLabel(item.category)}</span>
              <span className="category-nav-badges">
                {item.modifiedCount > 0 ? (
                  <span className="category-nav-badge badge-modified" title="Modified">
                    {item.modifiedCount}
                  </span>
                ) : null}
                {item.overriddenCount > 0 ? (
                  <span className="category-nav-badge badge-overridden" title="Overridden">
                    {item.overriddenCount}
                  </span>
                ) : null}
              </span>
            </button>
          );
        })}
      </div>

      {showSnapshots ? (
        <div className="category-nav-section">
          <div className="category-nav-divider" />
          <div className="category-nav-title">Config Snapshots</div>
          <button
            type="button"
            className="category-nav-item"
            onClick={onSnapshotsClick}
          >
            <span className="category-nav-icon">⊡</span>
            <span className="category-nav-label">Snapshots</span>
            {snapshotCount > 0 ? (
              <span className="category-nav-badges">
                <span className="category-nav-badge badge-count">{snapshotCount}</span>
              </span>
            ) : null}
          </button>
        </div>
      ) : null}
    </div>
  );
}
