"use client";

interface PendingChangesProps {
  changes: Record<string, unknown>;
}

export default function PendingChanges({ changes }: PendingChangesProps) {
  const entries = Object.entries(changes);
  if (!entries.length) {
    return (
      <div className="pending-changes">
        <h3>Pending Changes</h3>
        <p>No unsaved edits.</p>
      </div>
    );
  }

  return (
    <div className="pending-changes">
      <h3>Pending Changes</h3>
      <ul>
        {entries.map(([key, value]) => (
          <li key={key}>
            <strong>{key}</strong>: {JSON.stringify(value)}
          </li>
        ))}
      </ul>
    </div>
  );
}
