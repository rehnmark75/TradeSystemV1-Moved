export const WATCHLIST_DEFINITIONS: Record<
  string,
  { name: string; description: string; icon: string; type: "crossover" | "event"; tier: "primary" | "experimental" }
> = {
  ema_50_crossover: {
    name: "EMA 50 Crossover",
    description: "Price > EMA 200, Price crosses above EMA 50, Volume > 1M/day",
    icon: "📈",
    type: "crossover",
    tier: "primary"
  },
  ema_20_crossover: {
    name: "EMA 20 Crossover",
    description: "Price > EMA 200, Price crosses above EMA 20, Volume > 1M/day",
    icon: "📊",
    type: "crossover",
    tier: "experimental"
  },
  macd_bullish_cross: {
    name: "MACD Bullish Cross",
    description: "MACD crosses from negative to positive, Price > EMA 200, Volume > 1M/day",
    icon: "🔄",
    type: "crossover",
    tier: "experimental"
  },
  gap_up_continuation: {
    name: "Gap Up Continuation",
    description: "Gap up > 2% today, Closing above open, Price > EMA 200, Volume > 1M/day",
    icon: "🚀",
    type: "event",
    tier: "experimental"
  },
  rsi_oversold_bounce: {
    name: "RSI Oversold Bounce",
    description: "RSI(14) < 30, Price > EMA 200, Bullish candle, Volume > 1M/day",
    icon: "💪",
    type: "event",
    tier: "experimental"
  }
};
