export function getBasePath() {
  if (typeof window === "undefined") {
    return process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  }
  const nextData = (window as any).__NEXT_DATA__ ?? {};
  if (nextData.basePath || nextData.assetPrefix) {
    return nextData.basePath || nextData.assetPrefix || "";
  }
  const pathname = window.location?.pathname ?? "";
  if (pathname.startsWith("/trading/") || pathname === "/trading") {
    return "/trading";
  }
  return process.env.NEXT_PUBLIC_BASE_PATH ?? "";
}

export function apiUrl(path: string) {
  const basePath = getBasePath();
  if (!basePath) return path;
  return path.startsWith("/") ? `${basePath}${path}` : `${basePath}/${path}`;
}
