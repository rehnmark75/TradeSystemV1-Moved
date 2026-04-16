"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import EnvironmentToggle from "./EnvironmentToggle";

type NavItem = {
  href: string;
  label: string;
  shortLabel?: string;
};

type NavSection = {
  title: string;
  items: NavItem[];
};

const NAV_SECTIONS: NavSection[] = [
  {
    title: "Command",
    items: [
      { href: "/", label: "Mission Control", shortLabel: "Home" },
      { href: "/watchlists", label: "Equity Watchlists", shortLabel: "Watchlists" },
      { href: "/broker", label: "Broker Ledger", shortLabel: "Broker" },
      { href: "/market", label: "Market Regime", shortLabel: "Market" },
    ],
  },
  {
    title: "FX Stack",
    items: [
      { href: "/forex", label: "Forex Overview", shortLabel: "Forex" },
      { href: "/chart", label: "Execution Charting", shortLabel: "Chart" },
      { href: "/signals", label: "Signal Inspection", shortLabel: "Signals" },
    ],
  },
  {
    title: "Operations",
    items: [
      { href: "/infrastructure", label: "Infrastructure", shortLabel: "Infra" },
      { href: "/system", label: "System Pulse", shortLabel: "System" },
      { href: "/settings", label: "Strategy Settings", shortLabel: "Settings" },
    ],
  },
];

function isActivePath(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

function prettifySegment(segment: string): string {
  return segment
    .split("-")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function buildBreadcrumbs(pathname: string): string[] {
  const segments = pathname.split("/").filter(Boolean);
  if (segments.length === 0) return ["Mission Control"];
  return segments.map(prettifySegment);
}

function buildSectionTitle(pathname: string): string {
  for (const section of NAV_SECTIONS) {
    const active = section.items.find((item) => isActivePath(pathname, item.href));
    if (active) return active.label;
  }
  const crumbs = buildBreadcrumbs(pathname);
  return crumbs[crumbs.length - 1] ?? "Trading Workspace";
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const breadcrumbs = buildBreadcrumbs(pathname);
  const sectionTitle = buildSectionTitle(pathname);

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="app-brand-block">
          <Link href="/" className="app-brand-mark">
            <span className="app-brand-kicker">TradeSystem</span>
            <span className="app-brand-name">Operator Terminal</span>
          </Link>
          <p className="app-brand-copy">
            Institutional-grade monitoring for discretionary and automated execution.
          </p>
        </div>

        <nav className="app-nav" aria-label="Primary">
          {NAV_SECTIONS.map((section) => (
            <div key={section.title} className="app-nav-section">
              <div className="app-nav-title">{section.title}</div>
              <div className="app-nav-items">
                {section.items.map((item) => {
                  const active = isActivePath(pathname, item.href);
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`app-nav-link${active ? " active" : ""}`}
                    >
                      <span className="app-nav-link-main">{item.shortLabel ?? item.label}</span>
                      <span className="app-nav-link-sub">{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        <div className="app-sidebar-footer">
          <div className="app-sidebar-stat">
            <span>Stack</span>
            <strong>Trading UI</strong>
          </div>
          <div className="app-sidebar-stat">
            <span>Mode</span>
            <strong>Multi-Desk</strong>
          </div>
        </div>
      </aside>

      <div className="app-main">
        <header className="app-topbar">
          <div className="app-topbar-copy">
            <div className="app-breadcrumbs">
              {breadcrumbs.map((crumb, index) => (
                <span key={`${crumb}-${index}`} className="app-breadcrumb">
                  {crumb}
                </span>
              ))}
            </div>
            <div className="app-topbar-row">
              <div>
                <h1 className="app-topbar-title">{sectionTitle}</h1>
                <p className="app-topbar-subtitle">
                  Unified workspace for senior day traders, execution analysts, and bot operators.
                </p>
              </div>
              <div className="app-topbar-actions">
                <div className="app-status-card">
                  <span className="app-status-label">Session Focus</span>
                  <strong>Discretionary + Systematic</strong>
                </div>
                <EnvironmentToggle />
              </div>
            </div>
          </div>
        </header>

        <main className="app-workspace">{children}</main>
      </div>
    </div>
  );
}
