"use client";

import { useEnvironment } from "../lib/environment";

export default function EnvironmentToggle() {
  const { environment, setEnvironment } = useEnvironment();

  return (
    <div className="env-toggle">
      <button
        className={`env-btn ${environment === "demo" ? "env-active env-demo" : ""}`}
        onClick={() => setEnvironment("demo")}
      >
        DEMO
      </button>
      <button
        className={`env-btn ${environment === "live" ? "env-active env-live" : ""}`}
        onClick={() => setEnvironment("live")}
      >
        LIVE
      </button>
    </div>
  );
}
