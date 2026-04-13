"use client";

import ForexNav from "../_components/ForexNav";
import CandlestickChart from "../../../components/CandlestickChart";

export default function ForexChartPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0f172a" }}>
      <ForexNav activeHref="/forex/chart" />
      <div style={{ flex: 1, minHeight: 0 }}>
        <CandlestickChart />
      </div>
    </div>
  );
}
