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
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/chart">Chart</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/infrastructure">Infrastructure</Link>
          <Link href="/system">System Status</Link>
          <Link href="/settings">Settings</Link>
          <Link href="/signals" style={{ opacity: 0.4, fontSize: "0.8rem" }} title="Scanner debug view">Signals</Link>
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
              <p>EMA 50 crossover with Claude AI thesis, full-setup backtest (RS+DAQ), and quality filter.</p>
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

      <section className="landing-section">
        <div className="section-header">
          <h2>Operations</h2>
          <p>Infrastructure health, stream monitoring, and log intelligence.</p>
        </div>
        <div className="landing-grid">
          <Link href="/infrastructure" className="landing-card">
            <div className="landing-icon">🖥️</div>
            <div>
              <h3>Infrastructure</h3>
              <p>Container health, alerts, metrics, and restart actions via system-monitor.</p>
            </div>
          </Link>
          <Link href="/system" className="landing-card">
            <div className="landing-icon">📡</div>
            <div>
              <h3>System Status</h3>
              <p>Stream health, candle data freshness, operations feed, and log search.</p>
            </div>
          </Link>
          <Link href="/signals" className="landing-card" style={{ opacity: 0.55 }}>
            <div className="landing-icon">🔬</div>
            <div>
              <h3>Signals Debug</h3>
              <p>Raw scanner signal output for validation. Thesis and Claude analysis now shown in Watchlists.</p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  );
}
