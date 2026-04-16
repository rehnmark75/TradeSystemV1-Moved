import Link from "next/link";

const deskGroups = [
  {
    title: "Equity Desk",
    intro: "Long-bias watchlist discovery, broker oversight, and market regime tracking for discretionary stock operators.",
    cards: [
      {
        href: "/watchlists",
        title: "Watchlists",
        meta: "Discovery",
        body: "High-conviction scan boards with thesis context, validation state, and execution-ready detail panels.",
      },
      {
        href: "/broker",
        title: "Broker Ledger",
        meta: "Execution",
        body: "Performance, open exposure, and account telemetry framed like a professional dealing blotter.",
      },
      {
        href: "/market",
        title: "Market Regime",
        meta: "Context",
        body: "Breadth, sector rotation, and relative-strength leadership for top-down timing and capital allocation.",
      },
    ],
  },
  {
    title: "FX Desk",
    intro: "Scanner-driven workflow for live and demo FX operations, from market context to post-trade forensics.",
    cards: [
      {
        href: "/forex",
        title: "Forex Overview",
        meta: "Command",
        body: "Performance, strategy mix, and recent execution in one overview built for senior intraday operators.",
      },
      {
        href: "/chart",
        title: "Execution Charting",
        meta: "Monitoring",
        body: "Rejection overlays, trade markers, and context review for real-time scanner diagnostics.",
      },
      {
        href: "/signals",
        title: "Signal Inspection",
        meta: "Audit",
        body: "Raw scanner output and edge-case validation for systematic debugging and parameter review.",
      },
    ],
  },
  {
    title: "Platform Control",
    intro: "Infrastructure awareness and parameter governance so the trading stack feels like one coordinated system.",
    cards: [
      {
        href: "/infrastructure",
        title: "Infrastructure",
        meta: "Ops",
        body: "Containers, service health, alerts, and restart surfaces designed for fast operational decisions.",
      },
      {
        href: "/system",
        title: "System Pulse",
        meta: "Reliability",
        body: "Stream health, freshness checks, and platform events for monitoring data integrity under load.",
      },
      {
        href: "/settings",
        title: "Strategy Settings",
        meta: "Governance",
        body: "Centralized control of scanner, SMC, and trailing behavior with audit-friendly structure.",
      },
    ],
  },
];

export default function Page() {
  return (
    <div className="page">
      <section className="hero">
        <div className="mission-kicker">Institutional Command Surface</div>
        <h1>Built for senior day traders and bot operators who need signal clarity, not dashboard noise.</h1>
        <p>
          The platform is organized as a deliberate operating flow: discover opportunity, validate market context,
          inspect execution quality, then govern the stack with the same visual language across every desk.
        </p>
        <div className="mission-grid">
          <div className="mission-stat">
            <span>Flow</span>
            <strong>Discover → Validate → Execute → Audit</strong>
          </div>
          <div className="mission-stat">
            <span>Audience</span>
            <strong>Discretionary and systematic traders</strong>
          </div>
          <div className="mission-stat">
            <span>Standard</span>
            <strong>Commercial-grade operating environment</strong>
          </div>
        </div>
      </section>

      {deskGroups.map((group) => (
        <section key={group.title} className="landing-section">
          <div className="section-header">
            <div>
              <h2>{group.title}</h2>
              <p>{group.intro}</p>
            </div>
          </div>
          <div className="landing-grid">
            {group.cards.map((card) => (
              <Link href={card.href} className="landing-card" key={card.href}>
                <div className="landing-card-meta">{card.meta}</div>
                <div>
                  <h3>{card.title}</h3>
                  <p>{card.body}</p>
                </div>
                <span className="landing-card-cta">Open desk</span>
              </Link>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
