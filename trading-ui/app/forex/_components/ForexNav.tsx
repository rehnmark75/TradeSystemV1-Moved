import Link from "next/link";

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
  { href: "/forex/chart", label: "Chart" },
  { href: "/forex/strategy", label: "Strategy Performance" },
  { href: "/forex/trade-performance", label: "Trade Performance" },
  { href: "/forex/entry-timing", label: "Entry Timing" },
  { href: "/forex/mae-analysis", label: "MAE Analysis" },
  { href: "/forex/alert-history", label: "Alert History" },
  { href: "/forex/alert-data", label: "Alert Data" },
  { href: "/forex/trade-analysis", label: "Trade Analysis" },
  { href: "/forex/performance-snapshot", label: "Performance Snapshot" },
  { href: "/forex/market-intelligence", label: "Market Intelligence" },
  { href: "/forex/smc-rejections", label: "SMC Rejections" },
  { href: "/forex/validator-rejections", label: "Validator Rejections" },
  { href: "/forex/filter-effectiveness", label: "Filter Audit" }
];

export default function ForexNav({ activeHref, items = DEFAULT_ITEMS }: ForexNavProps) {
  return (
    <div className="forex-nav">
      {items.map((item) => {
        const isActive = activeHref === item.href;

        return (
          <Link
            key={item.href}
            href={item.href}
            className={`forex-pill${isActive ? " active" : ""}`}
            aria-current={isActive ? "page" : undefined}
          >
            {item.label}
          </Link>
        );
      })}
    </div>
  );
}
