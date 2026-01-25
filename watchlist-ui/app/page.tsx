import Link from "next/link";

export default function Page() {
  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Stocks Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
        </div>
      </div>

      <div className="hero">
        <h1>Stocks Command Center</h1>
        <p>Fast, focused views for watchlists and scanner signals.</p>
      </div>

      <div className="landing-grid">
        <Link href="/watchlists" className="landing-card">
          <div className="landing-icon">📋</div>
          <div>
            <h2>Watchlists</h2>
            <p>Virtualized list with deep technical + DAQ analysis and TradingView charts.</p>
          </div>
        </Link>
        <Link href="/signals" className="landing-card">
          <div className="landing-icon">📡</div>
          <div>
            <h2>Signals</h2>
            <p>Scanner signals with filters, Claude analysis, and trade plan context.</p>
          </div>
        </Link>
        <Link href="/broker" className="landing-card">
          <div className="landing-icon">💼</div>
          <div>
            <h2>Broker Stats</h2>
            <p>Account performance, open positions, and trade analytics synced from RoboMarkets.</p>
          </div>
        </Link>
        <Link href="/market" className="landing-card">
          <div className="landing-icon">🧭</div>
          <div>
            <h2>Market Context</h2>
            <p>Regime, breadth, sector rotation, and RS leadership in one view.</p>
          </div>
        </Link>
      </div>
    </div>
  );
}
