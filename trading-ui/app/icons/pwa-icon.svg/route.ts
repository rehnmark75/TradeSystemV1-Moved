const icon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="K.L.I.R.R icon"><rect width="512" height="512" rx="112" fill="#07111f"/><path d="M72 344h368" stroke="#203653" stroke-width="18" stroke-linecap="round"/><path d="M112 316l64-76 62 44 86-124 76 72" fill="none" stroke="#7cc7ff" stroke-width="26" stroke-linecap="round" stroke-linejoin="round"/><circle cx="112" cy="316" r="18" fill="#88ffd2"/><circle cx="176" cy="240" r="18" fill="#88ffd2"/><circle cx="238" cy="284" r="18" fill="#ffc86b"/><circle cx="324" cy="160" r="18" fill="#7cc7ff"/><circle cx="400" cy="232" r="18" fill="#ff6f91"/><text x="256" y="410" text-anchor="middle" font-family="Arial, sans-serif" font-size="64" font-weight="800" fill="#f4f7fb">KLR</text></svg>`;

export function GET() {
  return new Response(icon, {
    headers: {
      "Content-Type": "image/svg+xml",
      "Cache-Control": "public, max-age=86400"
    }
  });
}
