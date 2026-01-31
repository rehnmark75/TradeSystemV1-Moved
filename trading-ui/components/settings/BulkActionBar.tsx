"use client";

interface BulkActionBarProps {
  selectedCount: number;
  onCopyGlobal: () => void;
  onCopyPair: () => void;
  onReset: () => void;
}

export default function BulkActionBar({
  selectedCount,
  onCopyGlobal,
  onCopyPair,
  onReset
}: BulkActionBarProps) {
  return (
    <div className="bulk-action-bar">
      <span>{selectedCount} selected</span>
      <button onClick={onCopyGlobal}>Copy global → selected</button>
      <button onClick={onCopyPair}>Copy pair → selected</button>
      <button className="danger" onClick={onReset}>
        Reset selected
      </button>
    </div>
  );
}
