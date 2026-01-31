import { z } from "zod";

export const scannerUpdateSchema = z
  .object({
    scan_interval: z.number().int().min(30).max(600),
    min_confidence: z.number().min(0).max(1),
    default_timeframe: z.string().min(1),
    max_alerts_per_hour: z.number().int().min(1).max(200),
    max_alerts_per_epic_hour: z.number().int().min(1).max(50),
    max_open_positions: z.number().int().min(1).max(20),
    max_daily_trades: z.number().int().min(1).max(100),
    trading_start_hour: z.number().int().min(0).max(23),
    trading_end_hour: z.number().int().min(0).max(23),
    trading_cutoff_time_utc: z.number().int().min(0).max(23)
  })
  .partial()
  .passthrough();

export const smcUpdateSchema = z
  .object({
    fixed_stop_loss_pips: z.number().min(1).max(200),
    fixed_take_profit_pips: z.number().min(1).max(400),
    fixed_sl_tp_override_enabled: z.boolean()
  })
  .partial()
  .passthrough()
  .refine(
    (data) => {
      if (
        data.fixed_stop_loss_pips === undefined ||
        data.fixed_take_profit_pips === undefined
      ) {
        return true;
      }
      return data.fixed_take_profit_pips > data.fixed_stop_loss_pips;
    },
    {
      message: "TP must be greater than SL",
      path: ["fixed_take_profit_pips"]
    }
  );

export function validateScannerUpdates(updates: unknown) {
  return scannerUpdateSchema.safeParse(updates);
}

export function validateSmcUpdates(updates: unknown) {
  return smcUpdateSchema.safeParse(updates);
}
