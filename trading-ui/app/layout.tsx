import "./globals.css";
import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { EnvironmentProvider } from "../lib/environment";
import AppShell from "../components/AppShell";
import PwaRegistration from "../components/PwaRegistration";

const basePath = "/trading";

export const metadata: Metadata = {
  title: "K.L.I.R.R Trading Workspace",
  description: "Trading command workspace for watchlists, forex analytics, operations, and settings.",
  applicationName: "K.L.I.R.R",
  appleWebApp: {
    capable: true,
    title: "K.L.I.R.R",
    statusBarStyle: "black-translucent"
  },
  icons: {
    icon: [
      { url: `${basePath}/icons/pwa-icon.svg`, type: "image/svg+xml" }
    ],
    apple: [
      { url: `${basePath}/icons/pwa-icon.svg` }
    ]
  },
  formatDetection: {
    telephone: false
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0b1728"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href={`${basePath}/manifest.webmanifest`} crossOrigin="use-credentials" />
      </head>
      <body>
        <EnvironmentProvider>
          <AppShell>{children}</AppShell>
          <PwaRegistration />
        </EnvironmentProvider>
      </body>
    </html>
  );
}
