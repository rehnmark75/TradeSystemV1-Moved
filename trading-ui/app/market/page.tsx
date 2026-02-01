/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type RegimeData = {
  market_regime: string | null;
  spy_price: number | null;
  spy_vs_sma50_pct: number | null;
  spy_vs_sma200_pct: number | null;
  spy_trend: string | null;
  pct_above_sma200: number | null;
  pct_above_sma50: number | null;
  pct_above_sma20: number | null;
  new_highs_count: number | null;
  new_lows_count: number | null;
  high_low_ratio: number | null;
  advancing_count: number | null;
  declining_count: number | null;
  ad_ratio: number | null;
  avg_atr_pct: number | null;
  volatility_regime: string | null;
  recommended_strategies: Record<string, number> | null;
};

type SectorRow = {
  sector: string;
  sector_etf: string | null;
  rs_vs_spy: number | null;
  rs_percentile: number | null;
  rs_trend: string | null;
  sector_return_20d: number | null;
  sector_stage: string | null;
  stocks_in_sector: number | null;
  top_stocks: Array<{ ticker: string; rs_percentile: number; rs_trend?: string; price?: number | null }>;
};

type LeaderRow = {
  ticker: string;
  name: string | null;
  sector: string | null;
  current_price: number | null;
  rs_percentile: number | null;
  rs_trend: string | null;
  price_change_20d: number | null;
  trend_strength: string | null;
  pct_from_52w_high: number | null;
};

const toNumber = (value: unknown) => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const formatPct = (value: unknown, digits = 1) => {
  const num = toNumber(value);
  return num === null ? "N/A" : `${num.toFixed(digits)}%`;
};

const formatNumber = (value: unknown, digits = 1) => {
  const num = toNumber(value);
  return num === null ? "N/A" : num.toFixed(digits);
};

export default function MarketPage() {
  const apiPath = (path: string) => `../api/${path}`;
  const [regime, setRegime] = useState<RegimeData | null>(null);
  const [sectors, setSectors] = useState<SectorRow[]>([]);
  const [leaders, setLeaders] = useState<LeaderRow[]>([]);
  const [minRs, setMinRs] = useState(80);
  const [selectedSector, setSelectedSector] = useState<string>("");
  const [loadingLeaders, setLoadingLeaders] = useState(false);

  useEffect(() => {
    const load = async () => {
      const res = await fetch(`${apiPath("market/regime")}`);
      const data = await res.json();
      setRegime(data.row || null);
    };
    load();
  }, []);

  useEffect(() => {
    const load = async () => {
      const res = await fetch(`${apiPath("market/sectors")}`);
      const data = await res.json();
      const rows: SectorRow[] = data.rows || [];
      setSectors(rows);
      if (rows.length && !selectedSector) {
        setSelectedSector(rows[0].sector);
      }
    };
    load();
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoadingLeaders(true);
      const params = new URLSearchParams({ minRs: String(minRs), limit: "30" });
      const res = await fetch(`${apiPath("market/rs-leaders")}?${params.toString()}`);
      const data = await res.json();
      setLeaders(data.rows || []);
      setLoadingLeaders(false);
    };
    load();
  }, [minRs]);

  const regimeConfig = useMemo(() => {
    const status = regime?.market_regime || "unknown";
    const map: Record<string, { label: string; color: string; bg: string }> = {
      bull_confirmed: { label: "Bull Confirmed", color: "#1f8b4c", bg: "rgba(31, 139, 76, 0.12)" },
      bull_weakening: { label: "Bull Weakening", color: "#d89b24", bg: "rgba(216, 155, 36, 0.16)" },
      bear_weakening: { label: "Bear Weakening", color: "#d87124", bg: "rgba(216, 113, 36, 0.16)" },
      bear_confirmed: { label: "Bear Confirmed", color: "#c43d3d", bg: "rgba(196, 61, 61, 0.16)" },
      unknown: { label: "Unknown", color: "#6c6c6c", bg: "rgba(108, 108, 108, 0.12)" }
    };
    return map[status] || map.unknown;
  }, [regime]);

  const selectedSectorData = sectors.find((row) => row.sector === selectedSector);
  const sectorTone = (value: unknown) => {
    const num = toNumber(value);
    if (num === null) return "neutral";
    if (num > 1.1) return "strong";
    if (num > 1.0) return "good";
    if (num > 0.9) return "warn";
    return "bad";
  };

  const insights = useMemo(() => {
    const regimeStatus = (regime?.market_regime || "unknown").toLowerCase();
    const volatility = (regime?.volatility_regime || "unknown").toLowerCase();
    const isBull = regimeStatus.includes("bull");
    const isBear = regimeStatus.includes("bear");
    const isConfirmed = regimeStatus.includes("confirmed");
    const breadth = toNumber(regime?.pct_above_sma200);
    const adRatio = toNumber(regime?.ad_ratio);

    const badges: Array<{ label: string; tone: string }> = [];
    if (isBull && isConfirmed) badges.push({ label: "Trend OK", tone: "good" });
    if (isBear && isConfirmed) badges.push({ label: "Short Bias", tone: "bad" });
    if (volatility === "high") badges.push({ label: "Size Down", tone: "warn" });
    if (volatility === "low") badges.push({ label: "Breakout Watch", tone: "neutral" });
    if (breadth !== null && breadth > 60) badges.push({ label: "Breadth Strong", tone: "good" });
    if (breadth !== null && breadth < 40) badges.push({ label: "Breadth Weak", tone: "bad" });
    if (adRatio !== null && adRatio < 1) badges.push({ label: "Defensive Tone", tone: "warn" });

    const soWhat = isBull
      ? "Trend-following setups have the edge. Focus on leading sectors and pullback entries."
      : isBear
      ? "Risk-off tape. Favor defensive setups and tighten risk on counter-trend trades."
      : "Mixed regime. Keep position size smaller and wait for confirmation.";

    const setups = isBull
      ? ["Pullbacks to rising averages", "Breakout continuation", "Momentum in sector leaders"]
      : isBear
      ? ["Fade bear rallies", "Breakdown retests", "Defensive sector strength"]
      : ["Tight range reversals", "Wait for breakout trigger", "Selective leadership only"];

    const riskNotes = [
      volatility === "high" ? "Reduce size, wider stops" : "Normal size ok",
      breadth !== null && breadth < 40 ? "Avoid crowded longs" : "Breadth supportive",
      adRatio !== null && adRatio < 1 ? "Expect choppy tape" : "Adv/Decl supportive"
    ];

    const topSectors = [...sectors]
      .sort((a, b) => (toNumber(b.rs_vs_spy) ?? 0) - (toNumber(a.rs_vs_spy) ?? 0))
      .slice(0, 3)
      .map((row) => ({
        sector: row.sector,
        rs: formatNumber(row.rs_vs_spy, 2),
        trend: row.rs_trend || "stable"
      }));

    return { badges, soWhat, setups, riskNotes, topSectors };
  }, [regime, sectors]);

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
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Market Context</h1>
          <p>Regime, sector rotation, and relative strength leadership.</p>
        </div>
      </div>

      <div className="panel">
        <div className="market-intro">
          <div>
            <h2>So What?</h2>
            <p>{insights.soWhat}</p>
          </div>
          <div className="market-badges">
            {insights.badges.map((badge) => (
              <span key={badge.label} className={`market-badge badge-${badge.tone}`}>
                {badge.label}
              </span>
            ))}
          </div>
        </div>

        <div className="market-layout">
          <div className="market-left">
            <div className="market-grid">
              <div className="market-card">
                <div className="regime-card" style={{ borderColor: regimeConfig.color, background: regimeConfig.bg }}>
                  <div className="regime-title">{regimeConfig.label}</div>
                  <div className="regime-sub">
                    SPY: ${formatNumber(regime?.spy_price, 2)} ({formatNumber(regime?.spy_vs_sma200_pct, 1)}% vs SMA200)
                  </div>
                </div>
              </div>
              <div className="market-card">
                <div className="market-metric">
                  <span>SPY vs SMA50</span>
                  <strong>{formatPct(regime?.spy_vs_sma50_pct)}</strong>
                </div>
                <div className="market-metric">
                  <span>SPY Trend</span>
                  <strong>{regime?.spy_trend ? regime.spy_trend.toUpperCase() : "N/A"}</strong>
                </div>
                <div className="market-metric">
                  <span>Volatility</span>
                  <strong>{regime?.volatility_regime ? regime.volatility_regime.toUpperCase() : "N/A"}</strong>
                </div>
                <div className="market-metric">
                  <span>Avg ATR%</span>
                  <strong>{formatPct(regime?.avg_atr_pct, 1)}</strong>
                </div>
              </div>
              <div className="market-card">
                <h3>Strategy Weights</h3>
                <div className="strategy-grid">
                  {regime?.recommended_strategies
                    ? Object.entries(regime.recommended_strategies).map(([key, value]) => (
                        <div className="strategy-card" key={key}>
                          <span>{key.replace(/_/g, " ")}</span>
                          <strong>{Math.round(value * 100)}%</strong>
                        </div>
                      ))
                    : <div className="footer-note">No strategy data yet.</div>}
                </div>
              </div>
            </div>

            <div className="market-grid">
              <div className="market-card">
                <h3>Market Breadth</h3>
                <div className="breadth-grid">
                  <div className="breadth-card">
                    <span>% Above SMA200</span>
                    <strong>{formatPct(regime?.pct_above_sma200, 0)}</strong>
                  </div>
                  <div className="breadth-card">
                    <span>% Above SMA50</span>
                    <strong>{formatPct(regime?.pct_above_sma50, 0)}</strong>
                  </div>
                  <div className="breadth-card">
                    <span>New Highs/Lows</span>
                    <strong>{formatNumber(regime?.high_low_ratio, 2)}x</strong>
                    <em>{regime?.new_highs_count ?? 0}/{regime?.new_lows_count ?? 0}</em>
                  </div>
                  <div className="breadth-card">
                    <span>Adv/Decl</span>
                    <strong>{formatNumber(regime?.ad_ratio, 2)}</strong>
                    <em>{regime?.advancing_count ?? 0}/{regime?.declining_count ?? 0}</em>
                  </div>
                </div>
              </div>
              <div className="market-card">
                <h3>Sector Rotation</h3>
                <div className="sector-list">
                  {sectors.map((row) => {
                    const rsValue = toNumber(row.rs_vs_spy) ?? 1;
                    const barWidth = Math.min(140, Math.max(60, (rsValue - 0.8) * 100));
                    const tone = sectorTone(row.rs_vs_spy);
                    return (
                    <div
                      key={row.sector}
                      className={`sector-row ${selectedSector === row.sector ? "active" : ""} sector-${tone}`}
                      onClick={() => setSelectedSector(row.sector)}
                    >
                      <span>{row.sector}</span>
                      <div className="sector-bar">
                        <div
                          className={`sector-bar-fill sector-bar-${tone}`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <strong>{formatNumber(row.rs_vs_spy, 2)}</strong>
                    </div>
                    );
                  })}
                </div>
              </div>
              <div className="market-card">
                <h3>Top Stocks in {selectedSector || "Sector"}</h3>
                {selectedSectorData?.top_stocks?.length ? (
                  <div className="top-stock-grid">
                    {selectedSectorData.top_stocks.map((stock) => (
                      <div className="top-stock-card" key={stock.ticker}>
                        <strong>{stock.ticker}</strong>
                        <span>RS {Math.round(stock.rs_percentile)}%</span>
                        <em>{stock.rs_trend || "stable"}</em>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="footer-note">No top stocks data yet.</div>
                )}
              </div>
            </div>
          </div>

          <div className="market-right">
            <div className="market-card">
              <h3>Implications</h3>
              <div className="implication-block">
                <strong>Setups to Favor</strong>
                <ul>
                  {insights.setups.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="implication-block">
                <strong>Risk Adjustments</strong>
                <ul>
                  {insights.riskNotes.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="implication-block">
                <strong>Top Sectors</strong>
                <ul>
                  {insights.topSectors.map((item) => (
                    <li key={item.sector}>
                      {item.sector} <span>{item.rs}</span> <em>{item.trend}</em>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="market-card">
          <div className="rs-header">
            <h3>Relative Strength Leaders</h3>
            <div className="rs-filter">
              <label>Min RS %ile: {minRs}</label>
              <input
                type="range"
                min={50}
                max={95}
                step={5}
                value={minRs}
                onChange={(e) => setMinRs(Number(e.target.value))}
              />
            </div>
          </div>
          {loadingLeaders ? (
            <div className="footer-note">Loading RS leaders...</div>
          ) : leaders.length ? (
            <div className="leader-table">
              <div className="leader-row leader-header">
                <span>Ticker</span>
                <span>Sector</span>
                <span>Price</span>
                <span>RS %ile</span>
                <span>20D</span>
                <span>Trend</span>
              </div>
              {leaders.map((row) => (
                <div className="leader-row" key={row.ticker}>
                  <span className="ticker-btn">{row.ticker}</span>
                  <span>{row.sector || "-"}</span>
                  <span>{toNumber(row.current_price) !== null ? `$${formatNumber(row.current_price, 2)}` : "-"}</span>
                  <span>{toNumber(row.rs_percentile) !== null ? Math.round(toNumber(row.rs_percentile) ?? 0) : "-"}</span>
                  <span>{toNumber(row.price_change_20d) !== null ? `${formatNumber(row.price_change_20d, 1)}%` : "-"}</span>
                  <span>{row.trend_strength || "-"}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="footer-note">No RS leaders found.</div>
          )}
        </div>
      </div>
    </div>
  );
}
