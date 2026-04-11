import type { ReactNode } from "react";
import SettingsSidebar from "../../components/settings/SettingsSidebar";

export default function SettingsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="settings-shell">
      <SettingsSidebar />
      <div className="settings-content">{children}</div>
    </div>
  );
}
