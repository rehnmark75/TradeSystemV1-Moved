"use client";

type SlEvent = {
  time_iso: string;
  sl: number;
  event: string;
};

type Props = {
  slHistory: SlEvent[];
  openTime: number;
  closeTime: number;
  entry: number;
  direction: string;
  pipMult: number;
};

const EVENT_LABELS: Record<string, string> = {
  initial: "Order placed",
  early_be: "Early BE",
  break_even: "Breakeven",
  trail: "Trail move",
  stage1: "Stage 1 lock",
  stage2: "Stage 2 lock",
  stage3: "Stage 3 trail",
};

const fmtTime = (iso: string) => {
  const d = new Date(iso.includes("T") ? iso : iso.replace(" ", "T") + "Z");
  if (isNaN(d.valueOf())) return iso;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

const fmtSl = (sl: number) => sl.toFixed(5);

export default function TrailingTimeline({
  slHistory,
  openTime,
  closeTime,
  entry,
  direction,
  pipMult,
}: Props) {
  if (!slHistory.length) {
    return <div className="muted">No trailing events found in logs.</div>;
  }

  const pipsFromEntry = (sl: number) => {
    const raw = direction === "SELL" ? (entry - sl) * pipMult : (sl - entry) * pipMult;
    return raw;
  };

  return (
    <div className="trail-timeline">
      {slHistory.map((ev, i) => {
        const pips = pipsFromEntry(ev.sl);
        const isPositive = pips >= 0;
        const label = EVENT_LABELS[ev.event] ?? ev.event;
        const isLast = i === slHistory.length - 1;

        return (
          <div key={i} className={`trail-event ${isLast ? "trail-event-last" : ""}`}>
            <div className="trail-dot" />
            <div className="trail-line" />
            <div className="trail-content">
              <div className="trail-header">
                <span className="trail-label">{label}</span>
                <span className="trail-time">{fmtTime(ev.time_iso)}</span>
              </div>
              <div className="trail-detail">
                <span className="trail-sl">SL → {fmtSl(ev.sl)}</span>
                <span className={`trail-pips ${isPositive ? "pips-pos" : "pips-neg"}`}>
                  {isPositive ? "+" : ""}
                  {pips.toFixed(1)} pips from entry
                </span>
              </div>
            </div>
          </div>
        );
      })}

      <div className="trail-event trail-event-close">
        <div className="trail-dot trail-dot-close" />
        <div className="trail-content">
          <div className="trail-header">
            <span className="trail-label">Trade closed</span>
            <span className="trail-time">
              {fmtTime(new Date(closeTime * 1000).toISOString())}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
