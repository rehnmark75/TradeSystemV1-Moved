"use client";

interface DraftControlsProps {
  onRevert: () => void;
  onDiscard: () => void;
}

export default function DraftControls({ onRevert, onDiscard }: DraftControlsProps) {
  return (
    <div className="draft-controls">
      <button onClick={onRevert}>Revert to last saved</button>
      <button onClick={onDiscard}>Discard all pending changes</button>
    </div>
  );
}
