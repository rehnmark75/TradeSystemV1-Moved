import Link from "next/link";

const scannerCategories = [
  { key: "core", label: "Core" },
  { key: "indicators", label: "Indicators" },
  { key: "data-quality", label: "Data Quality" },
  { key: "trading-control", label: "Trading Control" },
  { key: "duplicate-detection", label: "Duplicate Detection" },
  { key: "risk-management", label: "Risk Management" },
  { key: "trading-hours", label: "Trading Hours" },
  { key: "order-executor", label: "Order Executor" },
  { key: "smc-conflict", label: "SMC Conflict" },
  { key: "claude-ai", label: "Claude AI" },
  { key: "audit", label: "Audit" }
];

export default function SettingsSidebar() {
  return (
    <nav className="settings-sidebar">
      <div className="settings-sidebar-section">
        <h3>Main</h3>
        <Link href="/">Home</Link>
        <Link href="/forex">Forex</Link>
        <Link href="/signals">Signals</Link>
        <Link href="/stocks">Stocks</Link>
        <Link href="/watchlists">Watchlists</Link>
        <Link href="/market">Market</Link>
        <Link href="/broker">Broker</Link>
      </div>
      <div className="settings-sidebar-section">
        <h3>Overview</h3>
        <Link href="/settings">Dashboard</Link>
        <Link href="/settings/audit">Audit Trail</Link>
      </div>
      <div className="settings-sidebar-section">
        <h3>Scanner</h3>
        {scannerCategories.map((category) => (
          <Link key={category.key} href={`/settings/scanner/${category.key}`}>
            {category.label}
          </Link>
        ))}
      </div>
      <div className="settings-sidebar-section">
        <h3>Strategy</h3>
        <Link href="/settings/strategy">SMC Global</Link>
        <Link href="/settings/strategy/pairs">Per-Pair Overrides</Link>
        <Link href="/settings/strategy/effective">Effective View</Link>
      </div>
    </nav>
  );
}
