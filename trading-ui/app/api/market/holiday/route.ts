import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export async function GET() {
  const client = await strategyConfigPool.connect();
  try {
    const result = await client.query(`
      SELECT condition_config
      FROM loss_prevention_rules
      WHERE is_enabled = TRUE
        AND condition_config->>'type' = 'date_block'
      ORDER BY id
    `);

    const today = new Date();
    const ymd = today.toISOString().slice(0, 10);
    const md = ymd.slice(5);

    for (const row of result.rows) {
      const cond = row.condition_config as {
        dates?: string[];
        month_days?: string[];
        label?: string;
      };
      const dates = cond.dates ?? [];
      const monthDays = cond.month_days ?? [];
      if (dates.includes(ymd) || monthDays.includes(md)) {
        return NextResponse.json({
          is_holiday: true,
          label: cond.label ?? "Bank holiday",
          date: ymd,
        });
      }
    }

    return NextResponse.json({ is_holiday: false, label: null, date: ymd });
  } catch {
    return NextResponse.json({ is_holiday: false, label: null, date: null });
  } finally {
    client.release();
  }
}
