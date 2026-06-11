import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

// SL/TP sweep grid, built per-cell as multipliers around that cell's own
// reference bracket. ref_grid(epic) is already instrument-scaled (gold ~80/160,
// FX ~10/15), so multiplying inherits the right pip scale automatically — no
// hard-coded FX grid that's too tight for gold.
const SL_MULT = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
const TP_MULT = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5];

// Round to instrument-appropriate "nice" pip steps: 1-pip resolution for FX-size
// brackets, coarser as the bracket grows so gold lands on 5/10-pip steps.
function niceRound(x: number): number {
  if (!Number.isFinite(x) || x <= 0) return 1;
  if (x < 10) return Math.max(1, Math.round(x));
  if (x < 30) return Math.round(x / 2) * 2;
  if (x < 100) return Math.round(x / 5) * 5;
  return Math.round(x / 10) * 10;
}

// Distinct, ascending axis from ref bracket × multipliers (drops collapsed dupes).
function buildAxis(ref: number, mults: number[]): number[] {
  const set = new Set<number>();
  for (const m of mults) {
    const v = niceRound(ref * m);
    if (v > 0) set.add(v);
  }
  return [...set].sort((a, b) => a - b);
}

// Cells thinner than this are flagged in the UI; forward-collected data is shallow
// and small samples mean-revert.
const THIN_N = 20;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

function normalizeSymbol(symbol: string | null) {
  if (!symbol) return "";
  return symbol
    .replace("CS.D.", "")
    .replace(".MINI.IP", "")
    .replace(".CEEM.IP", "")
    .replace(".CEE.IP", "");
}

const num = (v: unknown) => (v == null ? null : Number(v));
const round = (v: number, d = 2) => Math.round(v * 10 ** d) / 10 ** d;

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// A decided monitor-only signal carries its full MFE/MAE excursion. For any
// candidate (SL, TP) bracket we can re-derive the outcome:
//   - TP touched if MFE >= TP, SL touched if MAE >= SL.
//   - If both, order by time-to-extreme; ties resolve to SL (pessimistic, mirrors
//     the simulator's "SL-first within candle" rule).
//   - If neither, it's a timeout → net move at the 24h horizon.
// NOTE: timing is to the *max* excursion, not first-touch of the level, so the
// ordering is an approximation. Good enough to screen brackets, not to promote.
type SigPath = {
  mfe: number;
  mae: number;
  tMfe: number;
  tMae: number;
  net1440: number;
};

function bracketOutcome(s: SigPath, sl: number, tp: number): number {
  const hitTP = s.mfe >= tp;
  const hitSL = s.mae >= sl;
  if (hitTP && hitSL) {
    return s.tMae <= s.tMfe ? -sl : tp; // tie -> SL (pessimistic)
  }
  if (hitTP) return tp;
  if (hitSL) return -sl;
  return s.net1440; // timeout
}

type CellStats = { n: number; wins: number; pf: number | null; exp: number | null; wr: number | null };

function statsFromPnls(pnls: number[]): CellStats {
  const n = pnls.length;
  if (n === 0) return { n: 0, wins: 0, pf: null, exp: null, wr: null };
  let grossWin = 0;
  let grossLoss = 0;
  let wins = 0;
  let sum = 0;
  for (const p of pnls) {
    sum += p;
    if (p > 0) {
      grossWin += p;
      wins += 1;
    } else if (p < 0) {
      grossLoss += -p;
    }
  }
  return {
    n,
    wins,
    pf: grossLoss > 0 ? round(grossWin / grossLoss) : grossWin > 0 ? null : 0, // null = no losses (∞)
    exp: round(sum / n),
    wr: round((wins / n) * 100, 1),
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const env = searchParams.get("env") || "demo";
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    // Decided rows (ref_outcome set = TP/SL touched, or TIMEOUT at completion).
    // Crucially NOT filtered by status: a HIT_TP/HIT_SL row is decided the moment
    // it's touched but stays OPEN for 24h while MFE/MAE keep growing, so a
    // status='RESOLVED' filter would hide brand-new candidates entirely.
    const decidedResult = await forexPool.query(
      `
      SELECT
        strategy, epic, pair, direction, signal_timestamp,
        mfe_pips, mae_pips, early_mae_pips,
        time_to_mfe_minutes, time_to_mae_minutes,
        ref_sl_pips, ref_tp_pips, ref_outcome, ref_pnl_pips, pnl_1440m_pips
      FROM monitor_only_outcomes
      WHERE signal_timestamp >= $1
        AND environment = $2
        AND ref_outcome IS NOT NULL
      ORDER BY signal_timestamp DESC
      LIMIT 20000
      `,
      [since, env]
    );

    // Group decided rows by (strategy, epic) into candidate cells.
    type Cell = {
      strategy: string;
      epic: string;
      pair: string;
      refPnls: number[];
      mfe: number[];
      mae: number[];
      earlyMae: number[];
      loserMfe: number[];
      tp: number;
      sl: number;
      sl_set: Set<number>;
      tp_set: Set<number>;
      paths: SigPath[];
      tpHits: number;
      slHits: number;
      timeouts: number;
    };
    const cells = new Map<string, Cell>();

    for (const row of decidedResult.rows ?? []) {
      const strategy = row.strategy as string;
      const epic = row.epic as string;
      const key = `${strategy}__${epic}`;
      let cell = cells.get(key);
      if (!cell) {
        cell = {
          strategy,
          epic,
          pair: normalizeSymbol(epic) || (row.pair as string) || epic,
          refPnls: [],
          mfe: [],
          mae: [],
          earlyMae: [],
          loserMfe: [],
          tp: Number(row.ref_tp_pips ?? 0),
          sl: Number(row.ref_sl_pips ?? 0),
          sl_set: new Set(),
          tp_set: new Set(),
          paths: [],
          tpHits: 0,
          slHits: 0,
          timeouts: 0,
        };
        cells.set(key, cell);
      }
      const refPnl = Number(row.ref_pnl_pips ?? 0);
      const mfe = Number(row.mfe_pips ?? 0);
      const mae = Number(row.mae_pips ?? 0);
      cell.refPnls.push(refPnl);
      cell.mfe.push(mfe);
      cell.mae.push(mae);
      if (row.early_mae_pips != null) cell.earlyMae.push(Number(row.early_mae_pips));
      if (refPnl < 0) cell.loserMfe.push(mfe);
      if (row.ref_sl_pips != null) cell.sl_set.add(Number(row.ref_sl_pips));
      if (row.ref_tp_pips != null) cell.tp_set.add(Number(row.ref_tp_pips));
      if (row.ref_outcome === "HIT_TP") cell.tpHits += 1;
      else if (row.ref_outcome === "HIT_SL") cell.slHits += 1;
      else cell.timeouts += 1;
      cell.paths.push({
        mfe,
        mae,
        tMfe: row.time_to_mfe_minutes == null ? Number.POSITIVE_INFINITY : Number(row.time_to_mfe_minutes),
        tMae: row.time_to_mae_minutes == null ? Number.POSITIVE_INFINITY : Number(row.time_to_mae_minutes),
        net1440: Number(row.pnl_1440m_pips ?? 0),
      });
    }

    const avg = (arr: number[]) => (arr.length ? round(arr.reduce((a, b) => a + b, 0) / arr.length) : null);

    const candidates = Array.from(cells.values())
      .map((cell) => {
        const headline = statsFromPnls(cell.refPnls);
        const n = headline.n;

        const refSl = cell.sl_set.size === 1 ? [...cell.sl_set][0] : cell.sl;
        const refTp = cell.tp_set.size === 1 ? [...cell.tp_set][0] : cell.tp;

        // Per-cell, instrument-scaled SL/TP sweep grid.
        const slAxis = buildAxis(refSl || 10, SL_MULT);
        const tpAxis = buildAxis(refTp || 15, TP_MULT);
        const grid = slAxis.map((sl) =>
          tpAxis.map((tp) => {
            const st = statsFromPnls(cell.paths.map((p) => bracketOutcome(p, sl, tp)));
            return { sl, tp, pf: st.pf, exp: st.exp, wr: st.wr };
          })
        );
        // Best bracket = max expectancy (pips/trade); ties broken by higher PF.
        let best = grid[0][0];
        for (const rowg of grid) {
          for (const c of rowg) {
            const bExp = best.exp ?? -Infinity;
            const cExp = c.exp ?? -Infinity;
            if (cExp > bExp || (cExp === bExp && (c.pf ?? 0) > (best.pf ?? 0))) best = c;
          }
        }

        const deadOnArrival = cell.mfe.filter((m) => m < 2).length;

        return {
          strategy: cell.strategy,
          epic: cell.epic,
          pair: cell.pair,
          n,
          tp_hits: cell.tpHits,
          sl_hits: cell.slHits,
          timeouts: cell.timeouts,
          wr: headline.wr,
          pf: headline.pf,
          expectancy: headline.exp,
          // Scale-invariant edge: expectancy in units of risk (pips / SL pips), so
          // gold's large-pip cells rank fairly against tight FX cells.
          expectancy_r: headline.exp != null && refSl > 0 ? round(headline.exp / refSl, 3) : null,
          avg_mfe: avg(cell.mfe),
          avg_mae: avg(cell.mae),
          avg_early_mae: avg(cell.earlyMae),
          dead_on_arrival_pct: n ? round((deadOnArrival / n) * 100, 1) : null,
          median_mfe_losers: median(cell.loserMfe),
          ref_sl: refSl,
          ref_tp: refTp,
          per_month: round((n / days) * 30, 1),
          thin: n < THIN_N,
          sweep: { sl_values: slAxis, tp_values: tpAxis, grid, best },
        };
      })
      // Default order: R-multiple (scale-invariant) then PF; thin cells sink via a
      // penalty so a n=3 PF-infinity cell doesn't top the board. The UI can re-sort.
      .sort((a, b) => {
        const score = (c: typeof a) => (c.expectancy_r ?? -Infinity) - (c.thin ? 1000 : 0);
        return score(b) - score(a);
      });

    // Detail rows (most recent first, capped) — all statuses, for drill-down.
    const rowsResult = await forexPool.query(
      `
      SELECT
        alert_id, strategy, epic, pair, direction, signal_timestamp, status,
        entry_price, mfe_pips, mae_pips, early_mae_pips,
        pnl_60m_pips, pnl_240m_pips, pnl_1440m_pips,
        ref_sl_pips, ref_tp_pips, ref_outcome, ref_pnl_pips,
        time_to_mfe_minutes, candles_evaluated
      FROM monitor_only_outcomes
      WHERE signal_timestamp >= $1
        AND environment = $2
      ORDER BY signal_timestamp DESC
      LIMIT 1000
      `,
      [since, env]
    );

    const rows = (rowsResult.rows ?? []).map((row) => ({
      alert_id: Number(row.alert_id),
      strategy: row.strategy,
      epic: row.epic,
      pair: normalizeSymbol(row.epic) || row.pair,
      direction: row.direction,
      signal_timestamp: row.signal_timestamp,
      status: row.status,
      entry_price: num(row.entry_price),
      mfe_pips: num(row.mfe_pips),
      mae_pips: num(row.mae_pips),
      early_mae_pips: num(row.early_mae_pips),
      pnl_60m_pips: num(row.pnl_60m_pips),
      pnl_240m_pips: num(row.pnl_240m_pips),
      pnl_1440m_pips: num(row.pnl_1440m_pips),
      ref_sl_pips: num(row.ref_sl_pips),
      ref_tp_pips: num(row.ref_tp_pips),
      ref_outcome: row.ref_outcome,
      ref_pnl_pips: num(row.ref_pnl_pips),
      time_to_mfe_minutes: num(row.time_to_mfe_minutes),
      candles_evaluated: num(row.candles_evaluated),
    }));

    return NextResponse.json({ candidates, rows });
  } catch (error) {
    console.error("Failed to load monitor outcomes", error);
    return NextResponse.json({ error: "Failed to load monitor outcomes" }, { status: 500 });
  }
}
