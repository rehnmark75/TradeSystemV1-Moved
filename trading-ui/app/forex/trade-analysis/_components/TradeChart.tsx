"use client";

import { useEffect, useRef } from "react";
import type { IChartApi, ISeriesApi } from "lightweight-charts";

type Candle = { time: number; open: number; high: number; low: number; close: number };
type SlEvent = { time_iso: string; sl: number; event: string };
type MfeMae = { price: number | null; pips: number; time: number | null };

type Props = {
  candles: Candle[];
  entry: number;
  initialSl: number | null;
  tp: number | null;
  slHistory: SlEvent[];
  mfe: MfeMae;
  mae: MfeMae;
  openTime: number;
  closeTime: number;
  direction: string;
};

function isoToUnix(iso: string): number {
  const d = new Date(iso.includes("T") ? iso : iso.replace(" ", "T") + "Z");
  return isNaN(d.valueOf()) ? 0 : Math.floor(d.valueOf() / 1000);
}

export default function TradeChart({
  candles,
  entry,
  initialSl,
  tp,
  slHistory,
  mfe,
  mae,
  openTime,
  closeTime,
  direction,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current || !candles.length) return;

    let chart: IChartApi;
    let canceled = false;

    import("lightweight-charts").then(({ createChart, ColorType, LineStyle }) => {
      if (canceled || !containerRef.current) return;

      chart = createChart(containerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: "#0f172a" },
          textColor: "#94a3b8",
        },
        grid: {
          vertLines: { color: "#1e293b" },
          horzLines: { color: "#1e293b" },
        },
        crosshair: { mode: 1 },
        rightPriceScale: { borderColor: "#334155" },
        timeScale: { borderColor: "#334155", timeVisible: true, secondsVisible: false },
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      });

      // Candlestick series
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#22c55e",
        borderDownColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      });

      const sortedCandles = [...candles]
        .filter((c) => c.time > 0)
        .sort((a, b) => a.time - b.time);
      const seen = new Set<number>();
      const unique = sortedCandles.filter((c) => {
        if (seen.has(c.time)) return false;
        seen.add(c.time);
        return true;
      });
      candleSeries.setData(unique as any);
      candleSeriesRef.current = candleSeries;
      chartRef.current = chart;

      // Entry line
      const entrySeries = chart.addLineSeries({
        color: "#f59e0b",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false,
        title: "Entry",
      });
      if (unique.length) {
        entrySeries.setData([
          { time: unique[0].time as any, value: entry },
          { time: unique[unique.length - 1].time as any, value: entry },
        ]);
      }

      // TP line
      if (tp) {
        const tpSeries = chart.addLineSeries({
          color: "#22c55e",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          title: "TP",
        });
        if (unique.length) {
          tpSeries.setData([
            { time: unique[0].time as any, value: tp },
            { time: unique[unique.length - 1].time as any, value: tp },
          ]);
        }
      }

      // Initial SL line (dashed red)
      if (initialSl) {
        const slSeries = chart.addLineSeries({
          color: "#ef4444",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          priceLineVisible: false,
          lastValueVisible: false,
          title: "Orig SL",
        });
        if (unique.length) {
          slSeries.setData([
            { time: unique[0].time as any, value: initialSl },
            { time: unique[unique.length - 1].time as any, value: initialSl },
          ]);
        }
      }

      // SL step-line (solid red) showing actual SL over time
      if (slHistory.length) {
        const slLineSeries = chart.addLineSeries({
          color: "#f87171",
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          priceLineVisible: false,
          lastValueVisible: false,
          title: "SL",
        });

        const slPoints: { time: any; value: number }[] = [];
        for (const ev of slHistory) {
          const t = isoToUnix(ev.time_iso);
          if (t > 0) slPoints.push({ time: t, value: ev.sl });
        }
        slPoints.sort((a, b) => a.time - b.time);
        // Deduplicate by time (keep last value)
        const slDeduped = slPoints.filter((p, i) => i === slPoints.length - 1 || p.time !== slPoints[i + 1].time);
        // Extend the last SL point to close time
        if (slDeduped.length && closeTime > slDeduped[slDeduped.length - 1].time) {
          slDeduped.push({ time: closeTime, value: slDeduped[slDeduped.length - 1].value });
        }
        if (slDeduped.length) slLineSeries.setData(slDeduped);
      }

      // Markers: MFE, MAE, open, close
      const markers: any[] = [];

      if (mfe.time) {
        markers.push({
          time: mfe.time,
          position: direction === "SELL" ? "belowBar" : "aboveBar",
          color: "#22c55e",
          shape: "arrowDown",
          text: `MFE +${mfe.pips.toFixed(1)}p`,
          size: 1,
        });
      }
      if (mae.time && mae.pips > 0) {
        markers.push({
          time: mae.time,
          position: direction === "SELL" ? "aboveBar" : "belowBar",
          color: "#f97316",
          shape: "arrowUp",
          text: `MAE ${mae.pips.toFixed(1)}p`,
          size: 1,
        });
      }
      markers.push({
        time: openTime,
        position: "belowBar",
        color: "#60a5fa",
        shape: "arrowUp",
        text: "Open",
        size: 1,
      });
      markers.push({
        time: closeTime,
        position: "aboveBar",
        color: "#a78bfa",
        shape: "arrowDown",
        text: "Close",
        size: 1,
      });

      markers.sort((a, b) => a.time - b.time);
      candleSeries.setMarkers(markers);

      // Post-close shading via a separate area series
      const postCandles = unique.filter((c) => c.time >= closeTime);
      if (postCandles.length >= 2) {
        const shadeSeries = chart.addAreaSeries({
          topColor: "rgba(139,92,246,0.08)",
          bottomColor: "rgba(139,92,246,0.01)",
          lineColor: "rgba(139,92,246,0.15)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        shadeSeries.setData(
          postCandles.map((c) => ({ time: c.time as any, value: c.close }))
        );
      }

      // Fit to visible range
      chart.timeScale().fitContent();

      const ro = new ResizeObserver((entries) => {
        for (const e of entries) {
          chart.applyOptions({ width: e.contentRect.width, height: e.contentRect.height });
        }
      });
      ro.observe(containerRef.current!);
      (chart as any).__ro = ro;
    });

    return () => {
      canceled = true;
      if (chart) {
        (chart as any).__ro?.disconnect();
        chart.remove();
      }
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [candles, entry, initialSl, tp, slHistory, mfe, mae, openTime, closeTime, direction]);

  if (!candles.length) {
    return <div className="chart-placeholder">No candle data available for this trade.</div>;
  }

  return <div ref={containerRef} className="trade-chart-container" />;
}
