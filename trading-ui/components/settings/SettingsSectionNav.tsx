"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type SettingsNavItem = {
  href: string;
  label: string;
  description: string;
  eyebrow: string;
};

type SettingsNavGroup = {
  title: string;
  items: SettingsNavItem[];
};

const NAV_GROUPS: SettingsNavGroup[] = [
  {
    title: "Control",
    items: [
      {
        href: "/settings",
        label: "Overview",
        description: "Governance dashboard, health, and recent changes.",
        eyebrow: "Mission",
      },
      {
        href: "/settings/audit",
        label: "Audit Trail",
        description: "Change history, authorship, and control verification.",
        eyebrow: "Review",
      },
    ],
  },
  {
    title: "Execution",
    items: [
      {
        href: "/settings/scanner",
        label: "Scanner",
        description: "Discovery filters, validation thresholds, and scan behavior.",
        eyebrow: "Discovery",
      },
      {
        href: "/settings/strategy",
        label: "Strategy",
        description: "Global logic, pair overrides, and risk-sensitive tuning.",
        eyebrow: "Logic",
      },
    ],
  },
  {
    title: "Trade Handling",
    items: [
      {
        href: "/settings/trailing",
        label: "Trailing Stops",
        description: "Execution-stage trailing behavior by instrument and mode.",
        eyebrow: "Protection",
      },
      {
        href: "/settings/trailing-ratios",
        label: "Trailing Ratios",
        description: "Ratio ladders and progression tuning for managed exits.",
        eyebrow: "Ratios",
      },
    ],
  },
];

function isActivePath(pathname: string, href: string) {
  if (href === "/settings") return pathname === "/settings";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function SettingsSectionNav() {
  const pathname = usePathname();

  return (
    <section className="settings-nav-panel" aria-label="Settings navigation">
      <div className="settings-nav-header">
        <div>
          <div className="mission-kicker">Configuration Surface</div>
          <h2>Settings Control</h2>
        </div>
        <p>
          One governance rail for scanner logic, strategy behavior, audit visibility, and trailing controls.
        </p>
      </div>

      <div className="settings-nav-groups">
        {NAV_GROUPS.map((group) => (
          <div key={group.title} className="settings-nav-group">
            <div className="settings-nav-group-title">{group.title}</div>
            <div className="settings-nav-grid">
              {group.items.map((item) => {
                const active = isActivePath(pathname, item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`settings-nav-card${active ? " active" : ""}`}
                  >
                    <span className="settings-nav-card-eyebrow">{item.eyebrow}</span>
                    <strong>{item.label}</strong>
                    <span>{item.description}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
