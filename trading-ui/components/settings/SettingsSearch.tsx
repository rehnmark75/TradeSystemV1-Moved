"use client";

import { useEffect, useRef } from "react";

interface SettingsSearchProps {
  value: string;
  onChange: (value: string) => void;
}

export default function SettingsSearch({ value, onChange }: SettingsSearchProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="settings-search">
      <input
        ref={inputRef}
        type="search"
        placeholder="Search settings (Cmd/Ctrl + K)"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}
