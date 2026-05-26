"use client";

export type PostmortemData = {
  entry_verdict: "GOOD" | "AVERAGE" | "POOR";
  exit_verdict: "OPTIMAL" | "GOOD" | "PREMATURE" | "REVERSAL" | "ADVERSE";
  trailing_verdict: "EFFECTIVE" | "PREMATURE" | "MISSED_LOCK" | "NOT_TRIGGERED";
  entry_notes: string;
  exit_notes: string;
  trailing_notes: string;
  key_lesson: string;
  config_delta: {
    suggestion: string;
    rationale: string;
    pair: string;
    suggested_value: number | null;
  } | null;
  tags: string[];
  generated_at: string | null;
  agent_model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
};

type Props = {
  data: PostmortemData;
};

type VerdictColor = "green" | "amber" | "red" | "gray";

function verdictColor(verdict: string): VerdictColor {
  if (["GOOD", "EFFECTIVE", "OPTIMAL"].includes(verdict)) return "green";
  if (["AVERAGE", "PREMATURE", "MISSED_LOCK"].includes(verdict)) return "amber";
  if (["POOR", "REVERSAL", "ADVERSE"].includes(verdict)) return "red";
  return "gray";
}

function VerdictBadge({ label, verdict }: { label: string; verdict: string }) {
  const color = verdictColor(verdict);
  return (
    <div className={`pm-verdict pm-verdict-${color}`}>
      <span className="pm-verdict-label">{label}</span>
      <span className="pm-verdict-value">{verdict}</span>
    </div>
  );
}

function NoteBlock({ title, text }: { title: string; text: string }) {
  return (
    <div className="pm-note">
      <div className="pm-note-title">{title}</div>
      <div className="pm-note-text">{text}</div>
    </div>
  );
}

export default function PostmortemPanel({ data }: Props) {
  const hasDelta =
    data.config_delta && data.config_delta.suggestion !== "none";

  return (
    <div className="pm-panel">
      <div className="pm-header">
        <span className="pm-title">AI Post-Mortem</span>
        <span className="pm-meta">
          {data.agent_model ?? "claude-sonnet-4-6"}
          {data.input_tokens != null && (
            <> · {data.input_tokens + (data.output_tokens ?? 0)} tokens</>
          )}
          {data.generated_at && (
            <> · {new Date(data.generated_at).toLocaleString("sv-SE", { hour12: false })}</>
          )}
        </span>
      </div>

      {/* Verdicts row */}
      <div className="pm-verdicts">
        <VerdictBadge label="Entry" verdict={data.entry_verdict} />
        <VerdictBadge label="Exit" verdict={data.exit_verdict} />
        <VerdictBadge label="Trailing" verdict={data.trailing_verdict} />
      </div>

      {/* Notes */}
      <div className="pm-notes">
        <NoteBlock title="Entry" text={data.entry_notes} />
        <NoteBlock title="Exit" text={data.exit_notes} />
        <NoteBlock title="Trailing" text={data.trailing_notes} />
      </div>

      {/* Key lesson */}
      <div className="pm-lesson">
        <span className="pm-lesson-icon">💡</span>
        <span>{data.key_lesson}</span>
      </div>

      {/* Config delta — only shown when actionable */}
      {hasDelta && data.config_delta && (
        <div className="pm-delta">
          <div className="pm-delta-title">
            Config suggestion
            <span className="pm-delta-pair">{data.config_delta.pair}</span>
            <span className="pm-delta-action">{data.config_delta.suggestion}</span>
            {data.config_delta.suggested_value != null && (
              <span className="pm-delta-value">→ {data.config_delta.suggested_value}</span>
            )}
          </div>
          <div className="pm-delta-rationale">{data.config_delta.rationale}</div>
        </div>
      )}

      {/* Tags */}
      {data.tags && data.tags.length > 0 && (
        <div className="pm-tags">
          {data.tags.map((t) => (
            <span key={t} className="pm-tag">{t}</span>
          ))}
        </div>
      )}
    </div>
  );
}
