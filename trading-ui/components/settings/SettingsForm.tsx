"use client";
import PendingChanges from "./PendingChanges";
import DraftControls from "./DraftControls";
import ImpactPreview from "./ImpactPreview";

interface SettingsFormProps {
  title: string;
  changes: Record<string, unknown>;
  updatedBy: string;
  changeReason: string;
  onUpdatedByChange: (value: string) => void;
  onChangeReasonChange: (value: string) => void;
  onSave: (payload: { updatedBy: string; changeReason: string }) => void;
  onRevert: () => void;
  onDiscard: () => void;
}

export default function SettingsForm({
  title,
  changes,
  updatedBy,
  changeReason,
  onUpdatedByChange,
  onChangeReasonChange,
  onSave,
  onRevert,
  onDiscard
}: SettingsFormProps) {
  const criticalKeys = [
    "auto_trading_enabled",
    "enable_order_execution",
    "risk_per_trade_percent",
    "max_open_positions",
    "max_daily_trades"
  ];
  const hasCriticalChange = Object.keys(changes).some((key) =>
    criticalKeys.includes(key)
  );

  return (
    <div className="settings-form">
      <div className="settings-form-header">
        <h3>{title}</h3>
        <div className="settings-form-actions">
          <input
            placeholder="Updated by"
            value={updatedBy}
            onChange={(event) => onUpdatedByChange(event.target.value)}
          />
          <input
            placeholder="Change reason"
            value={changeReason}
            onChange={(event) => onChangeReasonChange(event.target.value)}
          />
          <button
            className="primary"
            disabled={
              !Object.keys(changes).length || !updatedBy.trim() || !changeReason.trim()
            }
            onClick={() => onSave({ updatedBy, changeReason })}
          >
            Save Changes
          </button>
        </div>
      </div>
      <DraftControls onRevert={onRevert} onDiscard={onDiscard} />
      {hasCriticalChange ? (
        <ImpactPreview level="warning" message="Critical setting change detected." />
      ) : (
        <ImpactPreview level="safe" message="No critical risk impacts detected." />
      )}
      <PendingChanges changes={changes} />
    </div>
  );
}
