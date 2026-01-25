import { NextResponse } from "next/server";
import { pool } from "../../../lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const ticker = searchParams.get("ticker");
  const context = searchParams.get("context");
  const limit = Number(searchParams.get("limit") || 20);

  if (!ticker) {
    return NextResponse.json({ error: "ticker is required" }, { status: 400 });
  }

  const params: Array<string | number> = [ticker.toUpperCase()];
  let where = `ticker = $1`;
  if (context) {
    params.push(context);
    where += ` AND context = $${params.length}`;
  }

  const client = await pool.connect();
  try {
    const query = `
      SELECT id, ticker, note_text, context, created_at, updated_at
      FROM stock_notes
      WHERE ${where}
      ORDER BY created_at DESC
      LIMIT ${limit}
    `;
    const result = await client.query(query, params);
    return NextResponse.json({ rows: result.rows });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load notes" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function POST(request: Request) {
  const body = await request.json();
  const ticker = String(body?.ticker || "").trim().toUpperCase();
  const note = String(body?.note || "").trim();
  const context = body?.context ? String(body.context).trim() : null;

  if (!ticker || !note) {
    return NextResponse.json({ error: "ticker and note are required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    const result = await client.query(
      `
      INSERT INTO stock_notes (ticker, note_text, context)
      VALUES ($1, $2, $3)
      RETURNING id, ticker, note_text, context, created_at, updated_at
      `,
      [ticker, note, context]
    );
    return NextResponse.json({ row: result.rows[0] });
  } catch (error) {
    return NextResponse.json({ error: "Failed to save note" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function PATCH(request: Request) {
  const body = await request.json();
  const id = Number(body?.id);
  const note = String(body?.note || "").trim();

  if (!id || !note) {
    return NextResponse.json({ error: "id and note are required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    const result = await client.query(
      `
      UPDATE stock_notes
      SET note_text = $1
      WHERE id = $2
      RETURNING id, ticker, note_text, context, created_at, updated_at
      `,
      [note, id]
    );
    return NextResponse.json({ row: result.rows[0] });
  } catch (error) {
    return NextResponse.json({ error: "Failed to update note" }, { status: 500 });
  } finally {
    client.release();
  }
}

export async function DELETE(request: Request) {
  const body = await request.json();
  const id = Number(body?.id);

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    await client.query(`DELETE FROM stock_notes WHERE id = $1`, [id]);
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json({ error: "Failed to delete note" }, { status: 500 });
  } finally {
    client.release();
  }
}
