"use client";

export interface TabDef { id: string; label: string; }

interface Props {
  tabs: TabDef[];
  active: string;
  onChange: (id: string) => void;
}

export default function SectionTabs({ tabs, active, onChange }: Props) {
  return (
    <div role="tablist" style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "20px" }}>
      {tabs.map(t => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === active}
          onClick={() => onChange(t.id)}
          className={`section-tab${t.id === active ? " active" : ""}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
