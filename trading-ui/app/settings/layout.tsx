import type { ReactNode } from "react";
import SettingsEnvBanner from "../../components/settings/SettingsEnvBanner";
import SettingsSectionNav from "../../components/settings/SettingsSectionNav";

export default function SettingsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="settings-shell">
      <div className="settings-content settings-content-wide">
        <SettingsEnvBanner />
        <SettingsSectionNav />
        {children}
      </div>
    </div>
  );
}
