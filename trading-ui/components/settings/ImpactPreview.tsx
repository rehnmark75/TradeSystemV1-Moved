"use client";

interface ImpactPreviewProps {
  level?: "safe" | "warning" | "danger";
  message?: string;
}

export default function ImpactPreview({ level = "safe", message }: ImpactPreviewProps) {
  return (
    <div className={`impact-badge ${level}`}>
      {message ?? "Impact preview unavailable"}
    </div>
  );
}
