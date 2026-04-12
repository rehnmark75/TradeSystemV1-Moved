"use client";
import { useRouter, useSearchParams } from "next/navigation";

export interface TabDef { id: string; label: string; }

interface Props {
  tabs: TabDef[];
  paramKey?: string;
}

export default function SectionTabs({ tabs, paramKey = "tab" }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const active = searchParams.get(paramKey) ?? tabs[0]?.id;

  const go = (id: string) => {
    const p = new URLSearchParams(searchParams.toString());
    p.set(paramKey, id);
    router.push(`?${p.toString()}`, { scroll: false });
  };

  return (
    <div role="tablist" style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "20px" }}>
      {tabs.map(t => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === active}
          onClick={() => go(t.id)}
          style={{
            padding: "7px 18px",
            borderRadius: "999px",
            border: "1px solid",
            borderColor: t.id === active ? "var(--accent)" : "var(--border)",
            background: t.id === active ? "var(--accent)" : "#fff7eb",
            color: t.id === active ? "#fff" : "#5b4a3a",
            fontWeight: 600,
            fontSize: "0.85rem",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
