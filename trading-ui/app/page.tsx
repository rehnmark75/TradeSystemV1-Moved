import Link from "next/link";

export default function Page() {
  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Trading Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="hero">
        <h1>Trading Command Center</h1>
        <p>Stocks, forex, and market context in one fast, focused workspace.</p>
      </div>

      <section className="landing-section">
        <div className="section-header">
          <h2>Stocks</h2>
          <p>Scanner intelligence, market context, and broker execution.</p>
        </div>
        <div className="landing-grid">
          <Link href="/watchlists" className="landing-card">
            <div className="landing-icon">📋</div>
            <div>
              <h3>Watchlists</h3>
              <p>Virtualized list with deep technical + DAQ analysis and TradingView charts.</p>
            </div>
          </Link>
          <Link href="/signals" className="landing-card">
            <div className="landing-icon">📡</div>
            <div>
              <h3>Signals</h3>
              <p>Scanner signals with filters, Claude analysis, and trade plan context.</p>
            </div>
          </Link>
          <Link href="/broker" className="landing-card">
            <div className="landing-icon">💼</div>
            <div>
              <h3>Broker Stats</h3>
              <p>Account performance, open positions, and trade analytics synced from RoboMarkets.</p>
            </div>
          </Link>
          <Link href="/market" className="landing-card">
            <div className="landing-icon">🧭</div>
            <div>
              <h3>Market Context</h3>
              <p>Regime, breadth, sector rotation, and RS leadership in one view.</p>
            </div>
          </Link>
        </div>
      </section>

      <section className="landing-section">
        <div className="section-header">
          <h2>Forex</h2>
          <p>Unified analytics for performance, strategy, and trade execution.</p>
        </div>
        <div className="landing-grid">
          <Link href="/forex" className="landing-card">
            <div className="landing-icon">📊</div>
            <div>
              <h3>Unified Analytics</h3>
              <p>Overview and analysis dashboards for the live trading book.</p>
            </div>
          </Link>
          <Link href="/settings" className="landing-card">
            <div className="landing-icon">🛠️</div>
            <div>
              <h3>Settings Center</h3>
              <p>Scanner + SMC configuration, overrides, and audit trail.</p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  );
}
