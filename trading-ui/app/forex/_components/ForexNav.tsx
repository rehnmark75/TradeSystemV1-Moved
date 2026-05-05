"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type ForexNavItem = {
  href: string;
  label: string;
};

type ForexNavProps = {
  activeHref?: string;
  items?: ForexNavItem[];
};

const DEFAULT_ITEMS: ForexNavItem[] = [
  { href: "/forex", label: "Overview" },
  { href: "/forex/backtests", label: "Backtests" },
  { href: "/forex/breakeven-optimizer", label: "Breakeven Optimizer" },
  { href: "/forex/chart", label: "Chart" },
  { href: "/forex/htf-analysis", label: "HTF Analysis" },
  { href: "/forex/strategy", label: "Strategy Performance" },
  { href: "/forex/trade-performance", label: "Trade Performance" },
  { href: "/forex/entry-timing", label: "Entry Timing" },
  { href: "/forex/mae-analysis", label: "MAE Analysis" },
  { href: "/forex/alert-history", label: "Alert History" },
  { href: "/forex/alert-data", label: "Alert Data" },
  { href: "/forex/trade-analysis", label: "Trade Analysis" },
  { href: "/forex/market-intelligence", label: "Market Intelligence" },
  { href: "/forex/economic-calendar", label: "Economic Calendar" },
  { href: "/forex/market-conditions", label: "Market Conditions" },
  { href: "/forex/adx-regime", label: "ADX Regime" },
  { href: "/forex/smc-rejections", label: "SMC Rejections" },
  { href: "/forex/strategy-rejections", label: "Strategy Rejections" },
  { href: "/forex/validator-rejections", label: "Validator Rejections" },
  { href: "/forex/claude-rejections", label: "Claude Rejections" },
  { href: "/forex/filter-effectiveness", label: "Filter Audit" }
];

export default function ForexNav({ activeHref, items = DEFAULT_ITEMS }: ForexNavProps) {
  const [open, setOpen] = useState(false);
  const activeItem = useMemo(
    () => items.find((item) => item.href === activeHref) ?? items[0],
    [activeHref, items]
  );

  useEffect(() => {
    setOpen(false);
  }, [activeHref]);

  return (
    <div className="forex-nav-shell">
      <button
        type="button"
        className="forex-mobile-menu-button"
        onClick={() => setOpen((next) => !next)}
        aria-expanded={open}
        aria-controls="forex-subpage-menu"
      >
        <span className="forex-mobile-menu-icon" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>
          <small>Forex pages</small>
          <strong>{activeItem?.label ?? "Overview"}</strong>
        </span>
      </button>

      <div id="forex-subpage-menu" className={`forex-nav${open ? " open" : ""}`}>
        {items.map((item) => {
          const isActive = activeHref === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`forex-pill${isActive ? " active" : ""}`}
              aria-current={isActive ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
