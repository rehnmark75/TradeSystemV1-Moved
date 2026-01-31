"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

interface HealthState {
  status: "connected" | "disconnected";
  checked_at: string;
  latency_ms: number;
}

export default function HealthIndicator() {
  const [health, setHealth] = useState<HealthState | null>(null);

  useEffect(() => {
    let mounted = true;
    const fetchHealth = async () => {
      try {
        const response = await fetch(apiUrl("/api/settings/health"));
        const data = await response.json();
        if (mounted) {
          setHealth(data);
        }
      } catch {
        if (mounted) {
          setHealth({
            status: "disconnected",
            checked_at: new Date().toISOString(),
            latency_ms: 0
          });
        }
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 60000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (!health) {
    return <div className="health-bar">Checking DB status...</div>;
  }

  return (
    <div className={`health-bar ${health.status}`}>
      <span>DB: {health.status}</span>
      <span>Latency: {health.latency_ms}ms</span>
      <span>Checked: {new Date(health.checked_at).toLocaleTimeString()}</span>
    </div>
  );
}
