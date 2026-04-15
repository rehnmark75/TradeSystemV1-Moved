import Link from "next/link";

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
        <Link href="/settings/scanner">Scanner Settings</Link>
        <Link href="/settings/scanner/audit">Audit Trail</Link>
      </div>
      <div className="settings-sidebar-section">
        <h3>Strategy</h3>
        <Link href="/settings/strategy">SMC Settings</Link>
        <Link href="/settings/strategy/effective">Effective View</Link>
      </div>
      <div className="settings-sidebar-section">
        <h3>Trailing Stops</h3>
        <Link href="/settings/trailing">Trailing Settings</Link>
        <Link href="/settings/trailing-ratios">Trailing Ratios</Link>
      </div>
    </nav>
  );
}
