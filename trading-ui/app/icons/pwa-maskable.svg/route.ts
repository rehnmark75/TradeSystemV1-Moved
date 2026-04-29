const icon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="K.L.I.R.R maskable icon"><rect width="512" height="512" fill="#07111f"/><circle cx="256" cy="256" r="220" fill="#0b1728"/><path d="M112 318l62-74 64 44 88-126 74 70" fill="none" stroke="#7cc7ff" stroke-width="30" stroke-linecap="round" stroke-linejoin="round"/><path d="M118 350h276" stroke="#203653" stroke-width="18" stroke-linecap="round"/><text x="256" y="414" text-anchor="middle" font-family="Arial, sans-serif" font-size="62" font-weight="800" fill="#f4f7fb">KLR</text></svg>`;

export function GET() {
  return new Response(icon, {
    headers: {
      "Content-Type": "image/svg+xml",
      "Cache-Control": "public, max-age=86400"
    }
  });
}
