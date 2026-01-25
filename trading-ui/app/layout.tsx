import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Watchlist Fast",
  description: "High-performance watchlist analysis"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
