"use client";

type TimeframeBadgesProps = {
  perf1w: number | null;
  perf1m: number | null;
  perf3m?: number | null;
};

export function TimeframeBadges({ perf1w, perf1m, perf3m }: TimeframeBadgesProps) {
  const toNumber = (val: number | null): number | null => {
    if (val === null || val === undefined) return null;
    const num = typeof val === 'string' ? parseFloat(val) : val;
    return isNaN(num) ? null : num;
  };

  const getSignal = (perf: number | null) => {
    const num = toNumber(perf);
    if (num === null) return "N/A";
    if (num >= 5) return "STRONG BUY";
    if (num >= 2) return "BUY";
    if (num >= -2) return "NEUTRAL";
    if (num >= -5) return "SELL";
    return "STRONG SELL";
  };

  const getSignalClass = (perf: number | null) => {
    const num = toNumber(perf);
    if (num === null) return "neutral";
    if (num >= 5) return "strong-buy";
    if (num >= 2) return "buy";
    if (num >= -2) return "neutral";
    if (num >= -5) return "sell";
    return "strong-sell";
  };

  const formatPerf = (perf: number | null) => {
    const num = toNumber(perf);
    if (num === null) return "-";
    return `${num >= 0 ? "+" : ""}${num.toFixed(1)}%`;
  };

  return (
    <div className="timeframe-badges">
      <div className={`tf-badge ${getSignalClass(perf1w)}`}>
        <span className="tf-label">Weekly</span>
        <span className="tf-value">{formatPerf(perf1w)}</span>
        <span className="tf-signal">{getSignal(perf1w)}</span>
      </div>

      <div className={`tf-badge ${getSignalClass(perf1m)}`}>
        <span className="tf-label">Monthly</span>
        <span className="tf-value">{formatPerf(perf1m)}</span>
        <span className="tf-signal">{getSignal(perf1m)}</span>
      </div>

      {perf3m !== null && perf3m !== undefined && (
        <div className={`tf-badge ${getSignalClass(perf3m)}`}>
          <span className="tf-label">3-Month</span>
          <span className="tf-value">{formatPerf(perf3m)}</span>
          <span className="tf-signal">{getSignal(perf3m)}</span>
        </div>
      )}

      <style jsx>{`
        .timeframe-badges {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin: 8px 0;
        }

        .tf-badge {
          display: flex;
          flex-direction: column;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid;
          min-width: 100px;
          font-size: 12px;
        }

        .tf-label {
          font-weight: 600;
          opacity: 0.8;
          margin-bottom: 4px;
        }

        .tf-value {
          font-size: 16px;
          font-weight: 700;
          margin-bottom: 2px;
        }

        .tf-signal {
          font-size: 10px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          opacity: 0.9;
        }

        .tf-badge.strong-buy {
          background: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);
          border-color: #2ecc71;
          color: #2ecc71;
        }

        .tf-badge.buy {
          background: linear-gradient(135deg, #1a3d47 0%, #0d1f28 100%);
          border-color: #3498db;
          color: #3498db;
        }

        .tf-badge.neutral {
          background: linear-gradient(135deg, #3d3d1a 0%, #28280d 100%);
          border-color: #f39c12;
          color: #f39c12;
        }

        .tf-badge.sell {
          background: linear-gradient(135deg, #472e1a 0%, #28180d 100%);
          border-color: #e67e22;
          color: #e67e22;
        }

        .tf-badge.strong-sell {
          background: linear-gradient(135deg, #471a1a 0%, #280d0d 100%);
          border-color: #e74c3c;
          color: #e74c3c;
        }

        @media (max-width: 768px) {
          .tf-badge {
            min-width: 80px;
            padding: 6px 10px;
          }

          .tf-value {
            font-size: 14px;
          }
        }
      `}</style>
    </div>
  );
}

// Compact version for list views
type TimeframeBadgesCompactProps = {
  perf1w: number | null;
  perf1m: number | null;
};

export function TimeframeBadgesCompact({ perf1w, perf1m }: TimeframeBadgesCompactProps) {
  const toNumber = (val: number | null): number | null => {
    if (val === null || val === undefined) return null;
    const num = typeof val === 'string' ? parseFloat(val) : val;
    return isNaN(num) ? null : num;
  };

  const getColor = (perf: number | null) => {
    const num = toNumber(perf);
    if (num === null) return "#888";
    if (num >= 2) return "#2ecc71";
    if (num >= -2) return "#f39c12";
    return "#e74c3c";
  };

  const formatPerf = (perf: number | null) => {
    const num = toNumber(perf);
    if (num === null) return "-";
    return `${num >= 0 ? "+" : ""}${num.toFixed(1)}%`;
  };

  return (
    <>
      <span style={{ color: getColor(perf1w), fontSize: '11px', fontWeight: '600' }}>
        {formatPerf(perf1w)}
      </span>
      <span style={{ color: getColor(perf1m), fontSize: '11px', fontWeight: '600' }}>
        {formatPerf(perf1m)}
      </span>
    </>
  );
}
