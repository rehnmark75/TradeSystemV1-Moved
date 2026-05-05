"use client";

export type CounterfactualVerdict = "SAVED" | "KILLED" | "NEUTRAL" | "UNKNOWN";

type Props = {
  verdict: CounterfactualVerdict;
  deltaP?: number | null;
  wouldHaveHit?: string | null;
  compact?: boolean;
};

export default function CounterfactualBadge({ verdict, deltaP, wouldHaveHit, compact }: Props) {
  const config: Record<CounterfactualVerdict, { label: string; cls: string }> = {
    SAVED: { label: "Trailing saved", cls: "cf-badge cf-saved" },
    KILLED: { label: "Trailing killed", cls: "cf-badge cf-killed" },
    NEUTRAL: { label: "Neutral", cls: "cf-badge cf-neutral" },
    UNKNOWN: { label: "Unknown", cls: "cf-badge cf-unknown" },
  };

  const { label, cls } = config[verdict] ?? config.UNKNOWN;

  if (compact) {
    return (
      <span className={cls} title={wouldHaveHit ? `Would have hit ${wouldHaveHit}` : undefined}>
        {label}
        {deltaP != null && verdict !== "NEUTRAL" && verdict !== "UNKNOWN"
          ? ` ${deltaP > 0 ? "+" : ""}${deltaP.toFixed(1)}p`
          : ""}
      </span>
    );
  }

  return (
    <div className={`${cls} cf-badge-full`}>
      <span className="cf-verdict">{label}</span>
      {deltaP != null && verdict !== "NEUTRAL" && verdict !== "UNKNOWN" && (
        <span className="cf-delta">
          {deltaP > 0 ? "+" : ""}
          {deltaP.toFixed(1)} pips
        </span>
      )}
      {wouldHaveHit && wouldHaveHit !== "neither" && (
        <span className="cf-detail">Would have hit {wouldHaveHit.replace("_", " ")}</span>
      )}
    </div>
  );
}
