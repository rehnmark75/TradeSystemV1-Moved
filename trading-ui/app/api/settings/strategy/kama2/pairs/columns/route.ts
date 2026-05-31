import { NextResponse } from "next/server";
import { getOverrideColumns } from "../../common";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const columns = (await getOverrideColumns())
      .filter((col) => !["id", "config_set", "epic", "created_at", "updated_at"].includes(col));
    return NextResponse.json({ columns });
  } catch (error) {
    console.error("Failed to load KAMA_V2 pair override columns", error);
    return NextResponse.json({ error: "Failed to load KAMA_V2 pair override columns" }, { status: 500 });
  }
}
