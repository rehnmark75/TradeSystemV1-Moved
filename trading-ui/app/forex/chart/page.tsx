"use client";

import ForexNav from "../_components/ForexNav";
import CandlestickChart from "../../../components/CandlestickChart";

export default function ForexChartPage() {
  return (
    <div className="forex-chart-page">
      <ForexNav activeHref="/forex/chart" />
      <div className="forex-chart-host">
        <CandlestickChart />
      </div>
    </div>
  );
}
