"use client";

import { useEnvironment } from "../../lib/environment";
import EnvironmentToggle from "../EnvironmentToggle";

export default function SettingsEnvBanner() {
  const { isLive } = useEnvironment();
  return (
    <div
      className={`settings-env-bar ${
        isLive ? "settings-env-bar-live" : "settings-env-bar-demo"
      }`}
    >
      <div className="settings-env-bar-label">
        <strong>Editing:</strong> {isLive ? "LIVE trading settings" : "DEMO trading settings"}
      </div>
      <EnvironmentToggle />
      {isLive && (
        <div className="settings-env-warning">
          ⚠️ Changes apply to live trading immediately. Verify before saving.
        </div>
      )}
    </div>
  );
}
