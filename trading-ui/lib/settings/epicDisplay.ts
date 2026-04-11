/**
 * Parse IG epic identifiers into human-readable pair names.
 * e.g. "CS.D.EURUSD.CEEM.IP" → "EUR/USD"
 */

const EPIC_OVERRIDES: Record<string, string> = {
  "CS.D.EURUSD.CEEM.IP": "EUR/USD",
  "CS.D.EURUSD.MINI.IP": "EUR/USD",
  "CS.D.GBPUSD.CEEM.IP": "GBP/USD",
  "CS.D.GBPUSD.MINI.IP": "GBP/USD",
  "CS.D.USDJPY.CEEM.IP": "USD/JPY",
  "CS.D.USDJPY.MINI.IP": "USD/JPY",
  "CS.D.AUDUSD.CEEM.IP": "AUD/USD",
  "CS.D.AUDUSD.MINI.IP": "AUD/USD",
  "CS.D.USDCHF.CEEM.IP": "USD/CHF",
  "CS.D.USDCHF.MINI.IP": "USD/CHF",
  "CS.D.USDCAD.CEEM.IP": "USD/CAD",
  "CS.D.USDCAD.MINI.IP": "USD/CAD",
  "CS.D.NZDUSD.CEEM.IP": "NZD/USD",
  "CS.D.NZDUSD.MINI.IP": "NZD/USD",
  "CS.D.EURJPY.CEEM.IP": "EUR/JPY",
  "CS.D.EURJPY.MINI.IP": "EUR/JPY",
  "CS.D.AUDJPY.CEEM.IP": "AUD/JPY",
  "CS.D.AUDJPY.MINI.IP": "AUD/JPY",
  "CS.D.GBPJPY.CEEM.IP": "GBP/JPY",
  "CS.D.GBPJPY.MINI.IP": "GBP/JPY",
};

/** Returns a short display name like "EUR/USD" for an IG epic */
export function epicToDisplayName(epic: string): string {
  if (EPIC_OVERRIDES[epic]) return EPIC_OVERRIDES[epic];

  // Generic fallback: extract the currency pair segment
  // CS.D.EURUSD.CEEM.IP → EURUSD → EUR/USD
  const parts = epic.split(".");
  if (parts.length >= 3) {
    const raw = parts[2]; // e.g. EURUSD
    if (raw.length === 6) {
      return `${raw.slice(0, 3)}/${raw.slice(3)}`;
    }
    return raw;
  }
  return epic;
}

/** Returns a short tag label like "EURUSD" for use in compact UI */
export function epicToTag(epic: string): string {
  const display = epicToDisplayName(epic);
  return display.replace("/", "");
}

/** Returns whether an epic is a JPY pair (needs different pip calculations) */
export function isJpyPair(epic: string): boolean {
  return epic.includes("JPY");
}

/** Returns whether an epic is a mini lot */
export function isMiniLot(epic: string): boolean {
  return epic.includes(".MINI.");
}
