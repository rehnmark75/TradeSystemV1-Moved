"use client";

import type { ReactNode } from "react";

interface SettingsGroupProps {
  title: string;
  description?: string;
  children: ReactNode;
}

export default function SettingsGroup({
  title,
  description,
  children
}: SettingsGroupProps) {
  return (
    <section className="settings-group">
      <div className="settings-group-header">
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
      <div className="settings-group-body">{children}</div>
    </section>
  );
}
