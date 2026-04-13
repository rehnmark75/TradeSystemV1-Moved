import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const configSet = searchParams.get("config_set") ?? "demo";

  try {
    const result = await strategyConfigPool.query(
      `
      SELECT po.epic
      FROM smc_simple_pair_overrides po
      JOIN smc_simple_global_config gc ON po.config_id = gc.id
      WHERE gc.is_active = TRUE
        AND gc.config_set = $1
        AND po.is_enabled = TRUE
      ORDER BY po.epic ASC
      `,
      [configSet]
    );

    const epics: string[] = result.rows.map((r) => r.epic as string);
    return NextResponse.json({ epics });
  } catch (error) {
    console.error("Failed to load epics", error);
    return NextResponse.json({ error: "Failed to load epics" }, { status: 500 });
  }
}
