const manifest = {
  name: "K.L.I.R.R Trading Workspace",
  short_name: "K.L.I.R.R",
  description: "Trading command workspace for watchlists, forex analytics, operations, and settings.",
  start_url: "/trading/",
  scope: "/trading/",
  display: "standalone",
  background_color: "#07111f",
  theme_color: "#0b1728",
  categories: ["finance", "productivity"],
  icons: [
    {
      src: "/trading/icons/pwa-icon.svg",
      sizes: "any",
      type: "image/svg+xml",
      purpose: "any"
    },
    {
      src: "/trading/icons/pwa-maskable.svg",
      sizes: "any",
      type: "image/svg+xml",
      purpose: "maskable"
    }
  ]
};

export function GET() {
  return Response.json(manifest, {
    headers: {
      "Cache-Control": "public, max-age=3600"
    }
  });
}
