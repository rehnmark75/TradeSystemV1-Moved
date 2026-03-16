import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * Proxy chart images from MinIO (internal Docker network) to the browser.
 * Usage: /api/forex/chart-image?url=http://minio:9000/claude-charts/...
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const imageUrl = searchParams.get("url");

  if (!imageUrl) {
    return NextResponse.json({ error: "Missing url parameter" }, { status: 400 });
  }

  // Only allow proxying from minio internal URLs
  if (!imageUrl.startsWith("http://minio:")) {
    return NextResponse.json({ error: "Invalid image source" }, { status: 403 });
  }

  try {
    const response = await fetch(imageUrl, { signal: AbortSignal.timeout(10000) });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Upstream returned ${response.status}` },
        { status: 502 }
      );
    }

    const contentType = response.headers.get("content-type") ?? "image/png";
    const buffer = await response.arrayBuffer();

    return new NextResponse(buffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400"
      }
    });
  } catch {
    return NextResponse.json({ error: "Failed to fetch chart image" }, { status: 502 });
  }
}
