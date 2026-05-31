"""
Inside-Day Breakout Strategy  (strategy_name: INSIDE_DAY)

The fresh "V2" replacement for the retired SMC_SIMPLE scalp. It is NOT SMC — a
May-2026 exhaustive search (7 SMC/directional variants) showed SMC scalping on FX
majors is edge-poor. This setup was the only one that survived multi-year OOS AND a
2-3x spread stress test:

    EURUSD PF 1.43 / USDJPY PF 1.42  (both 2020-22 and 2023-25 halves profitable;
    PF stays >1.2 even at triple spread because the ~50-70 pip targets dwarf cost).

Logic (low-frequency swing, ~2.5 trades/pair/month):
  1. Weekly momentum bias = position of the PRIOR completed week's close inside that
     week's range. Top 30% -> bullish-only; bottom 30% -> bearish-only; else stand aside.
  2. Inside day = the last COMPLETED daily candle sits strictly inside the prior day
     (high < prev high AND low > prev low) -> volatility compression. Range 10-100 pips.
  3. Entry = today's price breaks that inside day's high (bull) or low (bear), in the
     weekly-bias direction. One signal per pair per day.
  4. SL = opposite side of the inside day +/- 5% of daily ATR(14).  TP = 2.0R.

Design principles (lessons from v1's 7,891-line bloat):
  - No weighted confidence score (v1's was structurally anti-predictive).
  - No per-pair JSONB override sprawl; a handful of typed constants.
  - Single, explicit pipeline. EURUSD + USDJPY only. Monitor-only at launch.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .strategy_registry import register_strategy, StrategyInterface
except ImportError:  # pragma: no cover - dual-path import (see project memory)
    from forex_scanner.core.strategies.strategy_registry import register_strategy, StrategyInterface


# Pairs that cleared the OOS + spread-stress gate. Everything else stands aside.
_ENABLED = {
    "CS.D.EURUSD.CEEM.IP": {"pip": 0.0001},
    "CS.D.USDJPY.MINI.IP": {"pip": 0.01},
}

VERSION = "1.0.0"


@register_strategy("INSIDE_DAY")
class InsideDayBreakoutStrategy(StrategyInterface):
    # Tunables — kept tiny and explicit on purpose.
    WEEKLY_BIAS_Q = 0.30        # top/bottom 30% of weekly range -> directional bias
    ID_MIN_PIPS = 10.0          # inside-day range filter
    ID_MAX_PIPS = 100.0
    ATR_BUF_FRAC = 0.05         # SL buffer = 5% of daily ATR(14) beyond the ID extreme
    TP_R = 2.0                  # reward:risk
    ATR_PERIOD = 14
    MONITOR_ONLY = True         # launch posture — log signals, do not trade

    uses_smart_money_analysis = False

    def __init__(self, config=None, db_manager=None, logger=None,
                 config_override=None, **kwargs):
        # config_override is accepted for backtest-harness compatibility (the registry
        # passes it). v1 has no DB-overridable params (tunables are class constants),
        # so it is intentionally ignored — but must not raise.
        self.config = config
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = None              # injected by SignalDetector
        self._last_signal_day: Dict[str, str] = {}   # epic -> 'YYYY-MM-DD' (one/day)

    @property
    def strategy_name(self) -> str:
        return "INSIDE_DAY"

    def get_required_timeframes(self) -> List[str]:
        # 4h is resampled up to daily/weekly inside the strategy; 5m drives entry.
        return ["4h", "5m"]

    def reset_cooldowns(self) -> None:
        self._last_signal_day.clear()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _to_daily(df4h: pd.DataFrame) -> pd.DataFrame:
        d = df4h.copy()
        d["date"] = pd.to_datetime(d["start_time"]).dt.date
        agg = d.groupby("date").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"))
        agg.index = pd.to_datetime(agg.index)
        return agg

    def _daily_atr(self, daily: pd.DataFrame) -> pd.Series:
        pc = daily["close"].shift(1)
        tr = pd.concat([daily["high"] - daily["low"],
                        (daily["high"] - pc).abs(),
                        (daily["low"] - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(span=self.ATR_PERIOD, adjust=False).mean()

    def _weekly_bias(self, daily: pd.DataFrame, before) -> Optional[str]:
        """Bias from the PRIOR completed week (exclude the current, partial week)."""
        wk_start = before - pd.Timedelta(days=int(before.dayofweek))  # Monday of current week
        d = daily[daily.index < wk_start]
        if len(d) < 10:
            return None
        wk = d.groupby(d.index - pd.to_timedelta(d.index.dayofweek, unit="D")).agg(
            wh=("high", "max"), wl=("low", "min"), wc=("close", "last"))
        if len(wk) < 1:
            return None
        last = wk.iloc[-1]  # the most recent fully-completed week
        rng = last["wh"] - last["wl"]
        if rng <= 0:
            return None
        pos = (last["wc"] - last["wl"]) / rng
        if pos >= 1 - self.WEEKLY_BIAS_Q:
            return "bull"
        if pos <= self.WEEKLY_BIAS_Q:
            return "bear"
        return None

    # ── main ─────────────────────────────────────────────────────────────────
    def detect_signal(self, df_4h=None, df_5m=None, epic="", pair="", **kwargs) -> Optional[Dict]:
        try:
            cfg = _ENABLED.get(epic)
            if cfg is None:
                return None
            if df_4h is None or df_5m is None or len(df_4h) < 60 or len(df_5m) < 2:
                return None
            pip = cfg["pip"]

            # Normalise timestamps to tz-naive UTC. Live/backtest feeds (get_enhanced_data)
            # are tz-aware; the validation replay used tz-naive raw candles. Make every
            # downstream datetime comparison consistent so neither path raises.
            df_4h = df_4h.copy()
            df_5m = df_5m.copy()
            df_4h["start_time"] = pd.to_datetime(df_4h["start_time"], utc=True).dt.tz_localize(None)
            df_5m["start_time"] = pd.to_datetime(df_5m["start_time"], utc=True).dt.tz_localize(None)

            # current simulation/live time = last 5m bar
            now = pd.to_datetime(df_5m["start_time"].iloc[-1])
            today = now.normalize()
            day_key = today.strftime("%Y-%m-%d")
            if self._last_signal_day.get(epic) == day_key:
                return None  # already signalled today

            daily = self._to_daily(df_4h)
            completed = daily[daily.index < today]
            if len(completed) < 16:
                return None
            atr = self._daily_atr(daily)

            # today's running intraday extremes so far
            today_5m = df_5m[pd.to_datetime(df_5m["start_time"]).dt.normalize() == today]
            if len(today_5m) == 0:
                return None
            day_high = float(today_5m["high"].max())
            day_low = float(today_5m["low"].min())

            # Scan the last 3 completed days for an inside-day setup whose breakout
            # FIRST occurs today (the harness allows a ~3-day window after the inside day).
            # Entry is the breakout LEVEL itself (stop-order fill) -> deterministic and
            # independent of how late in the day the scan catches it.
            for k in range(1, 4):
                if len(completed) < k + 1:
                    break
                D = completed.iloc[-k]
                prev = completed.iloc[-(k + 1)]
                id_hi, id_lo = float(D["high"]), float(D["low"])
                if not (id_hi < float(prev["high"]) and id_lo > float(prev["low"])):
                    continue
                id_range_pips = (id_hi - id_lo) / pip
                if id_range_pips < self.ID_MIN_PIPS or id_range_pips > self.ID_MAX_PIPS:
                    continue
                bias = self._weekly_bias(daily, D.name)  # bias as of the INSIDE-DAY date
                if bias is None:
                    continue
                av = float(atr.loc[completed.index[-k]])
                if not np.isfinite(av) or av <= 0:
                    continue
                buf = av * self.ATR_BUF_FRAC
                # completed days strictly after the inside day (k=1 -> none yet)
                after = completed.iloc[-k + 1:] if k > 1 else completed.iloc[0:0]
                if bias == "bull":
                    if len(after) and bool((after["high"] > id_hi).any()):
                        continue  # already broke on an earlier day -> setup consumed
                    if day_high <= id_hi:
                        continue  # not broken yet today (still armed for a later day)
                    direction, brk, sl = "BUY", id_hi, id_lo - buf
                else:
                    if len(after) and bool((after["low"] < id_lo).any()):
                        continue
                    if day_low >= id_lo:
                        continue
                    direction, brk, sl = "SELL", id_lo, id_hi + buf

                entry = brk
                risk_pips = abs(entry - sl) / pip
                if risk_pips < 5 or risk_pips > 120:
                    continue
                reward_pips = risk_pips * self.TP_R
                tp = entry + reward_pips * pip if direction == "BUY" else entry - reward_pips * pip
                self._last_signal_day[epic] = day_key
                return self._build_signal(epic, pair, direction, entry, sl, tp,
                                          risk_pips, reward_pips, id_range_pips, av, bias, now)
            return None
        except Exception as exc:  # never let a strategy crash the scan loop
            self.logger.error(f"❌ [INSIDE_DAY] {epic}: {exc}")
            return None

    def _build_signal(self, epic, pair, direction, entry, sl, tp,
                      risk_pips, reward_pips, id_range_pips, atr, bias, ts) -> Dict:
        return {
            "signal": direction,
            "signal_type": direction,      # all three direction keys (TradeValidator + LPF)
            "direction": direction,
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": round(entry, 5),
            "stop_loss": round(sl, 5),
            "take_profit": round(tp, 5),
            "risk_pips": round(risk_pips, 1),      # BacktestScanner trade-eval keys
            "reward_pips": round(reward_pips, 1),
            "confidence_score": 0.65,              # fixed — NOT a quality ranker (v1 lesson)
            "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "entry_type": "INSIDE_DAY_BREAKOUT",
            "version": VERSION,
            "monitor_only": self.MONITOR_ONLY,
            "strategy_indicators": {
                "weekly_bias": bias,
                "inside_day_range_pips": round(id_range_pips, 1),
                "daily_atr": round(atr, 5),
            },
        }


def create_inside_day_strategy(config=None, db_manager=None, logger=None):
    return InsideDayBreakoutStrategy(config=config, db_manager=db_manager, logger=logger)
