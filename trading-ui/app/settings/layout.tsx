import type { ReactNode } from "react";
import SettingsSidebar from "../../components/settings/SettingsSidebar";
import SettingsEnvBanner from "../../components/settings/SettingsEnvBanner";

export default function SettingsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="settings-shell">
      <SettingsSidebar />
      <div className="settings-content">
        <SettingsEnvBanner />
        {children}
      </div>
    </div>
  );
}
