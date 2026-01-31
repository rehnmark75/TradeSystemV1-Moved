import { NextResponse } from "next/server";
import { strategyConfigPool } from "../../../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

async function loadActiveSmcConfig(client?: any) {
  const executor = client ?? strategyConfigPool;
  const result = await executor.query(
    `
      SELECT *
      FROM smc_simple_global_config
      WHERE is_active = TRUE
      ORDER BY updated_at DESC
      LIMIT 1
    `
  );
  return result.rows[0] ?? null;
}

async function loadOverride(client: any, configId: number, epic: string) {
  const result = await client.query(
    `
      SELECT *
      FROM smc_simple_pair_overrides
      WHERE config_id = $1 AND epic = $2
      LIMIT 1
    `,
    [configId, epic]
  );
  return result.rows[0] ?? null;
}

async function getOverrideColumns(client: any) {
  const result = await client.query(
    `
      SELECT column_name
      FROM information_schema.columns
      WHERE table_name = 'smc_simple_pair_overrides'
    `
  );
  return result.rows.map((row: { column_name: string }) => row.column_name);
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const { action, epics, source_epic, updated_by, change_reason } = body as {
    action?: string;
    epics?: string[];
    source_epic?: string;
    updated_by?: string;
    change_reason?: string;
  };

  if (!action || !Array.isArray(epics) || epics.length === 0) {
    return NextResponse.json(
      { error: "action and epics are required" },
      { status: 400 }
    );
  }
  if (!updated_by || !change_reason) {
    return NextResponse.json(
      { error: "updated_by and change_reason are required" },
      { status: 400 }
    );
  }

  const client = await strategyConfigPool.connect();
  try {
    await client.query("BEGIN");
    const globalConfig = await loadActiveSmcConfig(client);
    if (!globalConfig) {
      await client.query("ROLLBACK");
      return NextResponse.json(
        { error: "No active SMC config found" },
        { status: 404 }
      );
    }

    const affected: string[] = [];
    if (action === "reset") {
      await client.query(
        `
          DELETE FROM smc_simple_pair_overrides
          WHERE config_id = $1 AND epic = ANY($2)
        `,
        [globalConfig.id, epics]
      );
      affected.push(...epics);
    } else if (action === "copy-global") {
      await client.query(
        `
          DELETE FROM smc_simple_pair_overrides
          WHERE config_id = $1 AND epic = ANY($2)
        `,
        [globalConfig.id, epics]
      );
      affected.push(...epics);
    } else if (action === "copy-pair") {
      if (!source_epic) {
        await client.query("ROLLBACK");
        return NextResponse.json(
          { error: "source_epic is required for copy-pair" },
          { status: 400 }
        );
      }
      const sourceOverride = await loadOverride(client, globalConfig.id, source_epic);
      if (!sourceOverride) {
        await client.query("ROLLBACK");
        return NextResponse.json(
          { error: "Source override not found" },
          { status: 404 }
        );
      }
      const columns = await getOverrideColumns(client);
      const allowed = columns.filter(
        (column: string) =>
          ![
            "id",
            "config_id",
            "epic",
            "created_at",
            "updated_at",
            "updated_by",
            "change_reason"
          ].includes(column)
      );

      const overridePayload: Record<string, unknown> = {};
      allowed.forEach((column: string) => {
        overridePayload[column] = sourceOverride[column];
      });

      for (const epic of epics) {
        const columnNames = ["config_id", "epic", ...allowed, "updated_by", "change_reason"];
        const values = [
          globalConfig.id,
          epic,
          ...allowed.map((column: string) => overridePayload[column]),
          updated_by,
          change_reason
        ];
        const placeholders = columnNames.map((_, index) => `$${index + 1}`).join(", ");
        const updateSet = allowed
          .map((column: string) => `${column} = EXCLUDED.${column}`)
          .concat(["updated_by = EXCLUDED.updated_by", "change_reason = EXCLUDED.change_reason"])
          .join(", ");

        await client.query(
          `
            INSERT INTO smc_simple_pair_overrides (${columnNames.join(", ")})
            VALUES (${placeholders})
            ON CONFLICT (config_id, epic)
            DO UPDATE SET ${updateSet}
          `,
          values
        );
      }
      affected.push(...epics);
    } else {
      await client.query("ROLLBACK");
      return NextResponse.json({ error: "Unknown action" }, { status: 400 });
    }

    await client.query(
      `
        INSERT INTO smc_simple_config_audit
          (config_id, change_type, changed_by, change_reason, previous_values, new_values)
        VALUES ($1, 'BULK_UPDATE', $2, $3, $4, $5)
      `,
      [
        globalConfig.id,
        updated_by,
        change_reason,
        null,
        JSON.stringify({ action, epics: affected, source_epic })
      ]
    );

    await client.query("COMMIT");
    return NextResponse.json({ success: true, action, affected });
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to apply bulk override action", error);
    return NextResponse.json(
      { error: "Failed to apply bulk override action" },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
