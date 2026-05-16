"use client";

import { useState } from "react";
import CounterfactualBadge, { type CounterfactualVerdict } from "./CounterfactualBadge";

export type TradeRow = {
  id: number;
  symbol: string;
  direction: string;
  timestamp: string;
  closed_at: string | null;
  status: string;
  profit_loss: number | null;
  pnl_currency: string | null;
  pnl_display: string;
  entry_price: number | null;
  sl_price: number | null;
  initial_sl_price: number | null;
  tp_price: number | null;
  pips_gained: number | null;
  early_be_executed: boolean;
  stop_limit_changes_count: number;
  lifecycle_duration_minutes: number | null;
  is_scalp_trade: boolean;
  stages_reached: number;
  strategy: string | null;
  counterfactual?: { verdict: CounterfactualVerdict; delta_pips: number | null };
};

type SortKey = "timestamp" | "pips_gained" | "profit_loss" | "stop_limit_changes_count" | "stages_reached";

type Props = {
  trades: TradeRow[];
  selectedId: number | null;
  onSelect: (id: number) => void;
};

const shortEpic = (epic: string) =>
  epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "");

const fmtDate = (v: string) => {
  const d = new Date(v);
  return isNaN(d.valueOf())
    ? v
    : d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
};

const fmtPips = (v: number | null) =>
  v == null ? "-" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;

export default function TradeTable({ trades, selectedId, onSelect }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortAsc, setSortAsc] = useState(false);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((p) => !p);
    else { setSortKey(key); setSortAsc(false); }
  };

  const sorted = [...trades].sort((a, b) => {
    let av: number | string = a[sortKey] ?? (sortKey === "timestamp" ? "" : 0);
    let bv: number | string = b[sortKey] ?? (sortKey === "timestamp" ? "" : 0);
    if (av === bv) return 0;
    const cmp = av < bv ? -1 : 1;
    return sortAsc ? cmp : -cmp;
  });

  const th = (label: string, key: SortKey) => (
    <th
      className={`ta-th sortable ${sortKey === key ? "sort-active" : ""}`}
      onClick={() => toggleSort(key)}
    >
      {label}
      {sortKey === key ? (sortAsc ? " ▲" : " ▼") : ""}
    </th>
  );

  if (!trades.length) {
    return <div className="ta-empty">No trades match the current filters.</div>;
  }

  return (
    <div className="ta-table-wrap">
      <table className="ta-table">
        <thead>
          <tr>
            {th("Time", "timestamp")}
            <th className="ta-th">Pair</th>
            <th className="ta-th">Dir</th>
            <th className="ta-th">Strategy</th>
            <th className="ta-th">Entry</th>
            {th("Pips", "pips_gained")}
            {th("P&L", "profit_loss")}
            {th("Stages", "stages_reached")}
            {th("SL moves", "stop_limit_changes_count")}
            <th className="ta-th">Counterfactual</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((t) => {
            const isSelected = t.id === selectedId;
            const pip = t.pips_gained;
            const rowClass = [
              "ta-row",
              isSelected ? "ta-row-selected" : "",
              pip != null && pip > 0 ? "ta-row-win" : pip != null && pip < 0 ? "ta-row-loss" : "",
            ]
              .filter(Boolean)
              .join(" ");

            return (
              <tr key={t.id} className={rowClass} onClick={() => onSelect(t.id)}>
                <td className="ta-td">{fmtDate(t.timestamp)}</td>
                <td className="ta-td ta-pair">{shortEpic(t.symbol)}</td>
                <td className={`ta-td ta-dir ${t.direction === "BUY" ? "dir-buy" : "dir-sell"}`}>
                  {t.direction}
                </td>
                <td className="ta-td ta-strategy">{t.strategy ?? "-"}</td>
                <td className="ta-td ta-mono">
                  {t.entry_price != null ? t.entry_price.toFixed(5) : "-"}
                </td>
                <td className={`ta-td ta-mono ${pip != null && pip >= 0 ? "pips-pos" : "pips-neg"}`}>
                  {fmtPips(pip)}
                </td>
                <td className="ta-td ta-mono">{t.pnl_display}</td>
                <td className="ta-td ta-center">{t.stages_reached}</td>
                <td className="ta-td ta-center">{t.stop_limit_changes_count}</td>
                <td className="ta-td">
                  {t.counterfactual ? (
                    <CounterfactualBadge
                      verdict={t.counterfactual.verdict}
                      deltaP={t.counterfactual.delta_pips}
                      compact
                    />
                  ) : (
                    <span className="muted">–</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
