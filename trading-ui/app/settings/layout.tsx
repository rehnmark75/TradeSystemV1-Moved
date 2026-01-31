import type { ReactNode } from "react";
import SettingsSidebar from "../../components/settings/SettingsSidebar";

export default function SettingsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="settings-layout page">
      <SettingsSidebar />
      <div className="settings-content">{children}</div>
    </div>
  );
}
