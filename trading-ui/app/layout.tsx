import "./globals.css";
import type { ReactNode } from "react";
import { EnvironmentProvider } from "../lib/environment";
import AppShell from "../components/AppShell";

export const metadata = {
  title: "Watchlist Fast",
  description: "High-performance watchlist analysis"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <EnvironmentProvider>
          <AppShell>{children}</AppShell>
        </EnvironmentProvider>
      </body>
    </html>
  );
}
