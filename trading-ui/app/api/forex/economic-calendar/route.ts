import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const BASE_URL =
  process.env.ECONOMIC_CALENDAR_URL ?? "http://economic-calendar:8091/api/v1";

const CRITICAL_EVENTS = [
  "Non-Farm Employment Change",
  "NFP",
  "FOMC",
  "Federal Funds Rate",
  "ECB Press Conference",
  "Interest Rate Decision",
  "CPI",
  "Core CPI",
  "GDP",
  "Employment",
  "Unemployment"
];

const MONITORED_PAIRS = [
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "USDCHF",
  "AUDUSD",
  "NZDUSD",
  "USDCAD",
  "EURGBP",
  "EURJPY",
  "GBPJPY",
  "AUDJPY",
  "XAUUSD"
];

type ImpactLevel = "high" | "medium" | "low";

type CalendarEvent = {
  id: number | string;
  event_name: string;
  currency: string;
  event_date: string;
  impact_level: ImpactLevel;
  forecast_value?: string | null;
  previous_value?: string | null;
  actual_value?: string | null;
  time_until_event?: string | null;
  market_moving?: boolean | null;
};

type RiskBucket = {
  events_count: number;
  high_impact_events: Array<{ event: CalendarEvent; time_until: number }>;
  medium_impact_events: Array<{ event: CalendarEvent; time_until: number }>;
  critical_events: CalendarEvent[];
  nearest_event: { event: CalendarEvent; time_until: number } | null;
  highest_impact: "high" | "medium" | "low";
  risk_score: number;
  time_to_nearest_high_impact: number | null;
  currencies_affected: string[];
};

function parseHours(value: string | null) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 24;
  return Math.min(Math.max(Math.round(parsed), 1), 168);
}

function parsePair(value: string | null) {
  if (!value) return "EURUSD";
  const normalized = value.replace(/[^A-Za-z]/g, "").toUpperCase();
  if (normalized.length === 6) return normalized;
  if (normalized === "XAUUSD") return normalized;
  return "EURUSD";
}

function parseImpact(value: string | null) {
  if (value === "high" || value === "medium" || value === "all") return value;
  return "medium";
}

function extractCurrenciesFromPair(pair: string) {
  if (pair === "XAUUSD") return ["USD"];
  return [pair.slice(0, 3), pair.slice(3, 6)];
}

function isCriticalEvent(name: string) {
  return CRITICAL_EVENTS.some((critical) => name.includes(critical));
}

function minutesUntil(eventDate: string) {
  const eventTime = new Date(eventDate);
  if (Number.isNaN(eventTime.valueOf())) return null;
  return (eventTime.getTime() - Date.now()) / 60000;
}

function assessNewsRisk(events: CalendarEvent[]): RiskBucket {
  const risk: RiskBucket = {
    events_count: events.length,
    high_impact_events: [],
    medium_impact_events: [],
    critical_events: [],
    nearest_event: null,
    highest_impact: "low",
    risk_score: 0,
    time_to_nearest_high_impact: null,
    currencies_affected: []
  };

  const currencies = new Set<string>();

  for (const event of events) {
    const timeUntil = minutesUntil(event.event_date);
    if (timeUntil == null || timeUntil < 0) continue;

    currencies.add(event.currency);

    if (!risk.nearest_event || timeUntil < risk.nearest_event.time_until) {
      risk.nearest_event = { event, time_until: timeUntil };
    }

    const critical = isCriticalEvent(event.event_name);
    if (event.impact_level === "high" || critical) {
      risk.high_impact_events.push({ event, time_until: timeUntil });
      risk.highest_impact = "high";
      risk.time_to_nearest_high_impact =
        risk.time_to_nearest_high_impact == null
          ? timeUntil
          : Math.min(risk.time_to_nearest_high_impact, timeUntil);

      if (critical) {
        risk.critical_events.push(event);
      }
    } else if (event.impact_level === "medium") {
      risk.medium_impact_events.push({ event, time_until: timeUntil });
      if (risk.highest_impact !== "high") {
        risk.highest_impact = "medium";
      }
    }
  }

  let riskScore = 0;

  for (const highEvent of risk.high_impact_events) {
    const timeFactor = Math.max(0, (60 - highEvent.time_until) / 60);
    riskScore += 0.4 * timeFactor;
  }

  for (const mediumEvent of risk.medium_impact_events) {
    const timeFactor = Math.max(0, (30 - mediumEvent.time_until) / 30);
    riskScore += 0.2 * timeFactor;
  }

  if (risk.critical_events.length) {
    riskScore += 0.3;
  }

  risk.risk_score = Math.min(1, riskScore);
  risk.currencies_affected = Array.from(currencies);
  return risk;
}

function getPairState(risk: RiskBucket) {
  if (
    risk.time_to_nearest_high_impact != null &&
    risk.time_to_nearest_high_impact <= 30
  ) {
    return "blocked";
  }

  if (risk.nearest_event && risk.nearest_event.time_until <= 15) {
    return "caution";
  }

  if (risk.events_count > 0) {
    return "monitor";
  }

  return "clear";
}

function getPairHeadline(pair: string, risk: RiskBucket) {
  if (risk.critical_events.length && risk.time_to_nearest_high_impact != null) {
    const event = risk.critical_events[0];
    return `Critical ${event.currency} event in ${Math.round(
      risk.time_to_nearest_high_impact
    )}m`;
  }

  if (risk.nearest_event) {
    return `${pair} next event in ${Math.round(risk.nearest_event.time_until)}m`;
  }

  return `${pair} has no relevant events in range`;
}

function formatEventDay(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  });
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const hours = parseHours(searchParams.get("hours"));
  const pair = parsePair(searchParams.get("pair"));
  const impact = parseImpact(searchParams.get("impact"));

  try {
    const params = new URLSearchParams({
      hours: String(hours)
    });
    if (impact !== "all") {
      params.set("impact_level", impact);
    }

    const res = await fetch(`${BASE_URL}/events/upcoming?${params.toString()}`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000)
    });

    const data = await res.json();
    if (!res.ok) {
      return NextResponse.json(data, { status: res.status });
    }

    const events = ((data.upcoming_events ?? []) as CalendarEvent[])
      .map((event) => ({
        ...event,
        event_name: String(event.event_name ?? ""),
        currency: String(event.currency ?? "").toUpperCase(),
        impact_level: (event.impact_level ?? "low") as ImpactLevel
      }))
      .filter((event) => event.event_name && event.currency && event.event_date)
      .sort(
        (a, b) =>
          new Date(a.event_date).valueOf() - new Date(b.event_date).valueOf()
      );

    const selectedCurrencies = extractCurrenciesFromPair(pair);
    const selectedEvents = events.filter((event) =>
      selectedCurrencies.includes(event.currency)
    );
    const selectedRisk = assessNewsRisk(selectedEvents);

    const monitoredPairs = MONITORED_PAIRS.map((monitoredPair) => {
      const currencies = extractCurrenciesFromPair(monitoredPair);
      const pairEvents = events.filter((event) => currencies.includes(event.currency));
      const risk = assessNewsRisk(pairEvents);

      return {
        pair: monitoredPair,
        currencies,
        state: getPairState(risk),
        headline: getPairHeadline(monitoredPair, risk),
        events_count: risk.events_count,
        highest_impact: risk.highest_impact,
        risk_score: risk.risk_score,
        time_to_nearest_event: risk.nearest_event
          ? Math.round(risk.nearest_event.time_until)
          : null,
        time_to_nearest_high_impact:
          risk.time_to_nearest_high_impact == null
            ? null
            : Math.round(risk.time_to_nearest_high_impact)
      };
    }).sort((a, b) => b.risk_score - a.risk_score);

    const currencyTotals = events.reduce<Record<string, number>>((acc, event) => {
      acc[event.currency] = (acc[event.currency] ?? 0) + 1;
      return acc;
    }, {});

    const marketMovingCount = events.filter((event) => event.market_moving).length;
    const highImpactCount = events.filter(
      (event) => event.impact_level === "high" || isCriticalEvent(event.event_name)
    ).length;
    const mediumImpactCount = events.filter(
      (event) => event.impact_level === "medium"
    ).length;

    return NextResponse.json({
      generated_at: new Date().toISOString(),
      source: BASE_URL,
      query: {
        pair,
        hours,
        impact
      },
      summary: {
        total_events: events.length,
        high_impact_events: highImpactCount,
        medium_impact_events: mediumImpactCount,
        market_moving_events: marketMovingCount,
        currencies_affected: Object.keys(currencyTotals).length,
        next_event_at: events[0]?.event_date ?? null
      },
      selected_pair: {
        pair,
        currencies: selectedCurrencies,
        summary: {
          events_count: selectedRisk.events_count,
          highest_impact: selectedRisk.highest_impact,
          risk_score: selectedRisk.risk_score,
          nearest_event_at: selectedRisk.nearest_event?.event.event_date ?? null,
          nearest_event_name:
            selectedRisk.nearest_event?.event.event_name ?? null,
          time_to_nearest_event:
            selectedRisk.nearest_event == null
              ? null
              : Math.round(selectedRisk.nearest_event.time_until),
          time_to_nearest_high_impact:
            selectedRisk.time_to_nearest_high_impact == null
              ? null
              : Math.round(selectedRisk.time_to_nearest_high_impact),
          critical_events: selectedRisk.critical_events.length,
          affected_currencies: selectedRisk.currencies_affected
        },
        events: selectedEvents.map((event) => ({
          ...event,
          display_time: formatEventDay(event.event_date)
        }))
      },
      pair_risks: monitoredPairs,
      currencies: Object.entries(currencyTotals)
        .map(([currency, count]) => ({ currency, count }))
        .sort((a, b) => b.count - a.count),
      events: events.map((event) => ({
        ...event,
        display_time: formatEventDay(event.event_date),
        is_critical: isCriticalEvent(event.event_name)
      }))
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to load economic calendar: ${String(error)}` },
      { status: 502 }
    );
  }
}
