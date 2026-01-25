"use client";

import { useState } from "react";

export function PositionSizingCalculator() {
  const [accountSize, setAccountSize] = useState<string>("");
  const [riskPercent, setRiskPercent] = useState<string>("1");
  const [entryPrice, setEntryPrice] = useState<string>("");
  const [stopLoss, setStopLoss] = useState<string>("");
  const [showCalculator, setShowCalculator] = useState(false);

  const calculatePositionSize = () => {
    const account = parseFloat(accountSize);
    const risk = parseFloat(riskPercent);
    const entry = parseFloat(entryPrice);
    const stop = parseFloat(stopLoss);

    if (!account || !risk || !entry || !stop || entry <= stop) {
      return null;
    }

    const riskAmount = (account * risk) / 100;
    const riskPerShare = entry - stop;
    const shares = Math.floor(riskAmount / riskPerShare);
    const positionValue = shares * entry;
    const totalRisk = shares * riskPerShare;

    return {
      shares,
      positionValue,
      totalRisk,
      riskPerShare,
      positionPercent: (positionValue / account) * 100
    };
  };

  const result = calculatePositionSize();

  return (
    <div className="position-calculator">
      <button
        className="calculator-toggle"
        onClick={() => setShowCalculator(!showCalculator)}
      >
        📊 Position Size Calculator {showCalculator ? "▾" : "▸"}
      </button>

      {showCalculator && (
        <div className="calculator-content">
          <div className="calculator-grid">
            <div className="input-group">
              <label htmlFor="accountSize">Account Size ($)</label>
              <input
                id="accountSize"
                type="number"
                value={accountSize}
                onChange={(e) => setAccountSize(e.target.value)}
                placeholder="50000"
              />
            </div>

            <div className="input-group">
              <label htmlFor="riskPercent">Risk (%)</label>
              <input
                id="riskPercent"
                type="number"
                value={riskPercent}
                onChange={(e) => setRiskPercent(e.target.value)}
                placeholder="1.0"
                step="0.1"
              />
            </div>

            <div className="input-group">
              <label htmlFor="entryPrice">Entry Price ($)</label>
              <input
                id="entryPrice"
                type="number"
                value={entryPrice}
                onChange={(e) => setEntryPrice(e.target.value)}
                placeholder="150.00"
                step="0.01"
              />
            </div>

            <div className="input-group">
              <label htmlFor="stopLoss">Stop Loss ($)</label>
              <input
                id="stopLoss"
                type="number"
                value={stopLoss}
                onChange={(e) => setStopLoss(e.target.value)}
                placeholder="145.00"
                step="0.01"
              />
            </div>
          </div>

          {result && (
            <div className="calculator-results">
              <h4>📈 Position Details</h4>
              <div className="result-grid">
                <div className="result-item">
                  <span className="result-label">Shares:</span>
                  <span className="result-value">{result.shares.toLocaleString()}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Position Value:</span>
                  <span className="result-value">${result.positionValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Total Risk:</span>
                  <span className="result-value risk">${result.totalRisk.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Risk per Share:</span>
                  <span className="result-value">${result.riskPerShare.toFixed(2)}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Position %:</span>
                  <span className="result-value">{result.positionPercent.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          )}

          {!result && accountSize && riskPercent && entryPrice && stopLoss && (
            <div className="calculator-error">
              ⚠️ Check inputs: Entry must be greater than Stop Loss
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .position-calculator {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 20px;
        }

        .calculator-toggle {
          background: transparent;
          border: none;
          color: #e0e0e0;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          padding: 8px 0;
          width: 100%;
          text-align: left;
        }

        .calculator-toggle:hover {
          color: #4a9eff;
        }

        .calculator-content {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid #333;
        }

        .calculator-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
          margin-bottom: 20px;
        }

        .input-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .input-group label {
          font-size: 13px;
          color: #a0a0a0;
          font-weight: 500;
        }

        .input-group input {
          background: #0a0a0a;
          border: 1px solid #444;
          border-radius: 4px;
          padding: 10px 12px;
          color: #e0e0e0;
          font-size: 14px;
        }

        .input-group input:focus {
          outline: none;
          border-color: #4a9eff;
        }

        .calculator-results {
          background: #0f1419;
          border: 1px solid #2a3f5f;
          border-radius: 6px;
          padding: 16px;
        }

        .calculator-results h4 {
          margin: 0 0 16px 0;
          color: #4a9eff;
          font-size: 15px;
          font-weight: 600;
        }

        .result-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
        }

        .result-item {
          display: flex;
          justify-content: space-between;
          padding: 8px 12px;
          background: #1a1a1a;
          border-radius: 4px;
        }

        .result-label {
          font-size: 13px;
          color: #a0a0a0;
        }

        .result-value {
          font-size: 14px;
          font-weight: 600;
          color: #4a9eff;
        }

        .result-value.risk {
          color: #ff6b6b;
        }

        .calculator-error {
          padding: 12px;
          background: #3d1a1a;
          border: 1px solid #6b2929;
          border-radius: 4px;
          color: #ff6b6b;
          font-size: 13px;
        }

        @media (max-width: 768px) {
          .calculator-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
