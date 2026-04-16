import CandlestickChart from "../../components/CandlestickChart";

export default function ChartPage() {
  return (
    <div className="page">
      <div className="desk-intro">
        <div>
          <div className="mission-kicker">Execution Charting</div>
          <h2>Review trade markers and rejection paths on a chart surface built for fast forensic work.</h2>
          <p>
            Use this desk to understand how the system behaved in context: where structures failed, where Claude
            rejected, and how executed trades aligned with the live tape.
          </p>
        </div>
        <div className="desk-intro-meta">
          <div className="desk-intro-stat">
            <span>Use case</span>
            <strong>Trade review, rejection audit, scanner diagnostics</strong>
          </div>
          <div className="desk-intro-stat">
            <span>View</span>
            <strong>Chart-first operator workflow</strong>
          </div>
        </div>
      </div>
      <div className="panel full-height-panel">
        <CandlestickChart />
      </div>
    </div>
  );
}
