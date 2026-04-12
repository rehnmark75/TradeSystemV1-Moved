import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function POST(_req: Request, { params }: { params: { name: string } }) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/containers/${params.name}/restart`, {
      method: "POST", cache: "no-store", signal: AbortSignal.timeout(30000)
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
