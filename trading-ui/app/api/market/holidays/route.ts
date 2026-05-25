import { NextRequest, NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

export type UpcomingHoliday = {
  date: string;
  label: string;
  type: "fixed" | "recurring";
  days_until: number;
};

export async function GET(req: NextRequest) {
  const days = Math.min(365, Math.max(7, Number(req.nextUrl.searchParams.get("days") ?? "90")));

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
    today.setHours(0, 0, 0, 0);

    const upcoming: UpcomingHoliday[] = [];

    for (const row of result.rows) {
      const cond = row.condition_config as {
        dates?: string[];
        month_days?: string[];
        label?: string;
      };
      const label = cond.label ?? "Bank holiday";

      for (const dateStr of cond.dates ?? []) {
        const d = new Date(dateStr + "T00:00:00");
        const diffDays = Math.round((d.getTime() - today.getTime()) / 86400000);
        if (diffDays >= 0 && diffDays <= days) {
          upcoming.push({ date: dateStr, label, type: "fixed", days_until: diffDays });
        }
      }

      for (const md of cond.month_days ?? []) {
        for (const yearOffset of [0, 1]) {
          const dateStr = `${today.getFullYear() + yearOffset}-${md}`;
          const d = new Date(dateStr + "T00:00:00");
          if (isNaN(d.getTime())) continue;
          const diffDays = Math.round((d.getTime() - today.getTime()) / 86400000);
          if (diffDays >= 0 && diffDays <= days) {
            upcoming.push({ date: dateStr, label, type: "recurring", days_until: diffDays });
          }
        }
      }
    }

    upcoming.sort((a, b) => a.date.localeCompare(b.date));

    // Deduplicate by date (keep first occurrence)
    const seen = new Set<string>();
    const deduped = upcoming.filter((h) => {
      if (seen.has(h.date)) return false;
      seen.add(h.date);
      return true;
    });

    return NextResponse.json({ holidays: deduped, days });
  } catch {
    return NextResponse.json({ holidays: [], days });
  } finally {
    client.release();
  }
}
