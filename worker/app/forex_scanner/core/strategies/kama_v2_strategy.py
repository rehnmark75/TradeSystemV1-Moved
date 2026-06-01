#!/usr/bin/env python3
"""
KAMA V2 strategy — AUDUSD-only 5m adaptive-MA crossover.

Edge: KAMA(10,2,30) crossover during high-efficiency market conditions.
- Entry: price crosses KAMA with ER≥0.35, KAMA not counter-trending ≥0.5 pips
- EMA200 bias alignment
- MACD histogram sign confirmation
- RSI extreme rejection (avoid >70 BUY, <30 SELL)
- SL=10 pips, TP=15 pips (R:R 1.5)
- 30-min cooldown per epic
- monitor-only at launch

Research baseline (90d AUDUSD sweep, fixed SL/TP): n=93, WR=55.9%, PF=1.95, CI=[1.21, 2.76]

Unswept gates (default-off, ablate before enabling):
- session_filter / blocked_hours_utc  — sweep candidate: block 21-06 UTC
- adx_min                             — sweep candidate: 12–25
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy

logger = logging.getLogger(__name__)

_AUDUSD_EPIC = "CS.D.AUDUSD.MINI.IP"
_PIP = 0.0001

try:
    from forex_scanner.services.kama_v2_config_service import get_kama_v2_config_service
except ImportError:
    try:
        from services.kama_v2_config_service import get_kama_v2_config_service  # type: ignore
    except ImportError:
        get_kama_v2_config_service = None  # type: ignore

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from ...alerts.strategy_rejection_manager import StrategyRejectionManager  # type: ignore
    except ImportError:
        StrategyRejectionManager = None  # type: ignore


@dataclass
class KamaV2Config:
    # Pair routing
    # "" (empty) = DB-driven: accept any epic that is enabled in kama_v2_pair_overrides
    # Non-empty   = single-epic override (used by --override enabled_epic=X in backtests)
    enabled_epic: str = ""
    monitor_only: bool = True

    # KAMA / ER parameters (must match DataFetcher kama_period=10)
    kama_period: int = 10
    cross_er_min: float = 0.35      # ER threshold for crossover signals
    slope_bars: int = 3             # lookback bars for slope check
    slope_min_pips: float = 0.5     # reject if counter-slope exceeds this

    # Confirmation filters
    ema_trend: bool = True          # require price > EMA200 for BUY / < for SELL
    macd_filter: bool = True        # require MACD histogram sign alignment
    rsi_extreme_filter: bool = True # reject RSI>70 BUY, RSI<30 SELL

    # Session filter — default OFF until swept (ablate: session_filter=true + blocked_hours)
    session_filter: bool = False
    # Comma-separated UTC hours to block, e.g. "21,22,23,0,1,2,3" for rollover+dead-Asian
    blocked_hours_utc: str = "21,22,23,0,1,2,3"

    # ADX gate — default OFF until swept (ablate: adx_min=12 through 25)
    adx_min: float = 0.0            # 0.0 = no ADX gate

    # Risk
    fixed_stop_loss_pips: float = 10.0
    fixed_take_profit_pips: float = 15.0

    # Operational
    signal_cooldown_minutes: float = 30.0
    base_confidence: float = 0.60
    min_confidence: float = 0.60    # honest: base=0.60 so nothing below can pass
    max_confidence: float = 0.80    # honest: actual max achievable is ~0.80


def _apply_config_override(cfg: KamaV2Config, overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        try:
            if isinstance(current, bool):
                new_val = str(value).strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                new_val = int(float(value))
            elif isinstance(current, float):
                new_val = float(value)
            else:
                new_val = value
            setattr(cfg, key, new_val)
        except Exception as exc:
            logger.warning("KAMA_V2 override failed for %s=%r: %s", key, value, exc)


@register_strategy("KAMA_V2")
class KamaV2Strategy(StrategyInterface):
    """KAMA crossover with ER gate, EMA200 bias, MACD + RSI confirmation."""

    def __init__(
        self,
        config: Optional[KamaV2Config] = None,
        logger=None,
        db_manager=None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self._config_override = config_override

        self.config = config or KamaV2Config()
        _apply_config_override(self.config, self._config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        # Config service for multi-pair DB-driven mode (None = not available)
        self._cfg_svc = get_kama_v2_config_service() if get_kama_v2_config_service else None
        self._rej_mgr = StrategyRejectionManager("KAMA_V2", db_manager) if StrategyRejectionManager else None

        self.logger.info(
            "[KAMA_V2] initialized | epic=%s monitor_only=%s "
            "cross_er=%.2f slope_bars=%d slope_min=%.1f pips "
            "EMA200=%s MACD=%s RSI=%s SL/TP=%.0f/%.0f cooldown=%.0fm "
            "session_filter=%s adx_min=%.1f",
            self.config.enabled_epic,
            self.config.monitor_only,
            self.config.cross_er_min,
            self.config.slope_bars,
            self.config.slope_min_pips,
            self.config.ema_trend,
            self.config.macd_filter,
            self.config.rsi_extreme_filter,
            self.config.fixed_stop_loss_pips,
            self.config.fixed_take_profit_pips,
            self.config.signal_cooldown_minutes,
            self.config.session_filter,
            self.config.adx_min,
        )

    # -------------------------------------------------------------------------
    # StrategyInterface
    # -------------------------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "KAMA_V2"

    def get_required_timeframes(self):
        return ["5m"]

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()

    def get_config(self) -> KamaV2Config:
        return self.config

    # -------------------------------------------------------------------------
    # Signal detection
    # -------------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        epic: Optional[str] = None,
        pair: Optional[str] = None,
        spread_pips: float = 1.5,
        current_timestamp: Optional[datetime] = None,
        routing_context: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate the last completed 5m bar for a KAMA crossover setup."""
        try:
            cfg = self.config
            pair_key = epic or pair or "UNKNOWN"
            self._current_timestamp = current_timestamp

            # ── Epic gate ─────────────────────────────────────────────────────
            # Single-epic override mode (backtest): enabled_epic is set explicitly
            if cfg.enabled_epic:
                if epic != cfg.enabled_epic:
                    return None
            else:
                # DB-driven multi-pair mode: check kama_v2_pair_overrides
                if self._cfg_svc is None or not self._cfg_svc.is_pair_enabled(epic):
                    return None

            # ── Per-pair param resolution ─────────────────────────────────────
            # DB overrides win; fall back to global KamaV2Config for unset columns.
            # In single-epic override mode (backtest), use global config values as-is.
            svc = self._cfg_svc if (self._cfg_svc and not cfg.enabled_epic) else None
            sl_pips       = svc.get_pair_sl_pips(pair_key, cfg.fixed_stop_loss_pips)   if svc else cfg.fixed_stop_loss_pips
            tp_pips       = svc.get_pair_tp_pips(pair_key, cfg.fixed_take_profit_pips) if svc else cfg.fixed_take_profit_pips
            cross_er_min  = svc.get_pair_cross_er_min(pair_key, cfg.cross_er_min)      if svc else cfg.cross_er_min
            adx_min       = svc.get_pair_adx_min(pair_key, cfg.adx_min)                if svc else cfg.adx_min
            sess_filter   = svc.get_pair_session_filter(pair_key, cfg.session_filter)  if svc else cfg.session_filter
            blocked_hours = svc.get_pair_blocked_hours(pair_key, cfg.blocked_hours_utc) if svc else cfg.blocked_hours_utc
            monitor_only  = svc.is_monitor_only(pair_key)                               if svc else cfg.monitor_only

            # ── Data check ───────────────────────────────────────────────────
            min_rows = cfg.kama_period + cfg.slope_bars + 5
            if df_trigger is None or len(df_trigger) < min_rows:
                self._reject(pair_key, pair, "INSUFFICIENT_DATA", f"need {min_rows} rows, got {len(df_trigger) if df_trigger is not None else 0}")
                return None

            required_cols = ["close", "kama_10", "kama_10_er"]
            for col in required_cols:
                if col not in df_trigger.columns:
                    self._reject(pair_key, pair, "MISSING_INDICATOR", f"missing column {col}")
                    return None

            # ── Timestamp ────────────────────────────────────────────────────
            if current_timestamp is not None:
                eval_ts = current_timestamp
                if not isinstance(eval_ts, datetime):
                    eval_ts = pd.Timestamp(eval_ts).to_pydatetime()
                if eval_ts.tzinfo is None:
                    eval_ts = eval_ts.replace(tzinfo=timezone.utc)
            else:
                if "start_time" in df_trigger.columns:
                    raw = df_trigger["start_time"].iloc[-1]
                    raw_ts = pd.Timestamp(raw)
                    eval_ts = raw_ts.to_pydatetime() if not pd.isna(raw_ts) else datetime.now(timezone.utc)
                    if eval_ts.tzinfo is None:
                        eval_ts = eval_ts.replace(tzinfo=timezone.utc)
                else:
                    eval_ts = datetime.now(timezone.utc)
            self._current_timestamp = eval_ts  # type: ignore[assignment]

            # ── Session filter ───────────────────────────────────────────────
            if sess_filter and blocked_hours:
                try:
                    blocked = {int(h.strip()) for h in blocked_hours.split(",") if h.strip()}
                    if eval_ts.hour in blocked:
                        self._reject(pair_key, pair, "SESSION", f"hour={eval_ts.hour} UTC blocked", hour_utc=eval_ts.hour)
                        return None
                except Exception:
                    pass

            # ── Cooldown ─────────────────────────────────────────────────────
            if pair_key in self._cooldowns:
                elapsed = (eval_ts - self._cooldowns[pair_key]).total_seconds() / 60
                if elapsed < cfg.signal_cooldown_minutes:
                    self._reject(
                        pair_key,
                        pair,
                        "COOLDOWN",
                        f"{elapsed:.1f} / {cfg.signal_cooldown_minutes:.0f} min",
                        hour_utc=eval_ts.hour,
                        details={"elapsed_minutes": elapsed, "cooldown_minutes": cfg.signal_cooldown_minutes},
                    )
                    return None

            # ── Extract series ────────────────────────────────────────────────
            close = df_trigger["close"]
            kama = df_trigger["kama_10"]
            er = df_trigger["kama_10_er"]

            # Need at least 2 rows for crossover detection
            if len(close) < 2:
                return None

            close_now = float(close.iloc[-1])
            close_prev = float(close.iloc[-2])
            kama_now = float(kama.iloc[-1])
            kama_prev = float(kama.iloc[-2])
            er_now = float(er.iloc[-1]) if not pd.isna(er.iloc[-1]) else 0.0

            # ── ER gate ───────────────────────────────────────────────────────
            if er_now < cross_er_min:
                self._reject(
                    pair_key,
                    pair,
                    "ER_GATE",
                    f"ER {er_now:.3f} < {cross_er_min:.3f}",
                    hour_utc=eval_ts.hour,
                    details={"er": er_now, "cross_er_min": cross_er_min},
                )
                return None

            # ── Crossover detection ───────────────────────────────────────────
            crossed_up = (close_now > kama_now) and (close_prev <= kama_prev)
            crossed_down = (close_now < kama_now) and (close_prev >= kama_prev)

            if not crossed_up and not crossed_down:
                return None

            # ── Slope check (counter-slope rejection) ─────────────────────────
            slope_idx = max(0, len(kama) - 1 - cfg.slope_bars)
            kama_old = float(kama.iloc[slope_idx])
            kama_delta_pips = (kama_now - kama_old) / _PIP

            if crossed_up and kama_delta_pips <= -cfg.slope_min_pips:
                self._reject(
                    pair_key,
                    pair,
                    "COUNTER_SLOPE",
                    f"BUY KAMA counter-slope {kama_delta_pips:.2f} pips",
                    direction="BUY",
                    hour_utc=eval_ts.hour,
                    details={"kama_delta_pips": kama_delta_pips, "slope_min_pips": cfg.slope_min_pips},
                )
                return None
            if crossed_down and kama_delta_pips >= cfg.slope_min_pips:
                self._reject(
                    pair_key,
                    pair,
                    "COUNTER_SLOPE",
                    f"SELL KAMA counter-slope {kama_delta_pips:.2f} pips",
                    direction="SELL",
                    hour_utc=eval_ts.hour,
                    details={"kama_delta_pips": kama_delta_pips, "slope_min_pips": cfg.slope_min_pips},
                )
                return None

            direction = "BUY" if crossed_up else "SELL"

            # ── ADX gate ──────────────────────────────────────────────────────
            if adx_min > 0 and "adx" in df_trigger.columns:
                adx_val = float(df_trigger["adx"].iloc[-1])
                if not pd.isna(adx_val) and adx_val < adx_min:
                    self._reject(
                        pair_key,
                        pair,
                        "ADX_GATE",
                        f"{direction} ADX {adx_val:.1f} < {adx_min:.1f}",
                        direction=direction,
                        hour_utc=eval_ts.hour,
                        details={"adx": adx_val, "adx_min": adx_min},
                    )
                    return None

            # ── EMA200 alignment ──────────────────────────────────────────────
            if cfg.ema_trend and "ema_200" in df_trigger.columns:
                ema200 = float(df_trigger["ema_200"].iloc[-1])
                if not pd.isna(ema200):
                    if direction == "BUY" and close_now <= ema200:
                        self._reject(
                            pair_key,
                            pair,
                            "EMA200_ALIGNMENT",
                            "BUY price below EMA200",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"close": close_now, "ema_200": ema200},
                        )
                        return None
                    if direction == "SELL" and close_now >= ema200:
                        self._reject(
                            pair_key,
                            pair,
                            "EMA200_ALIGNMENT",
                            "SELL price above EMA200",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"close": close_now, "ema_200": ema200},
                        )
                        return None

            # ── MACD histogram sign ───────────────────────────────────────────
            if cfg.macd_filter and "macd_histogram" in df_trigger.columns:
                macd_hist = float(df_trigger["macd_histogram"].iloc[-1])
                if not pd.isna(macd_hist):
                    if direction == "BUY" and macd_hist <= 0:
                        self._reject(
                            pair_key,
                            pair,
                            "MACD_ALIGNMENT",
                            f"BUY MACD histogram {macd_hist:.5f} <= 0",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"macd_histogram": macd_hist},
                        )
                        return None
                    if direction == "SELL" and macd_hist >= 0:
                        self._reject(
                            pair_key,
                            pair,
                            "MACD_ALIGNMENT",
                            f"SELL MACD histogram {macd_hist:.5f} >= 0",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"macd_histogram": macd_hist},
                        )
                        return None

            # ── RSI extreme rejection ─────────────────────────────────────────
            if cfg.rsi_extreme_filter and "rsi" in df_trigger.columns:
                rsi_val = float(df_trigger["rsi"].iloc[-1])
                if not pd.isna(rsi_val):
                    if direction == "BUY" and rsi_val >= 70:
                        self._reject(
                            pair_key,
                            pair,
                            "RSI_EXTREME",
                            f"BUY RSI {rsi_val:.1f} >= 70",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"rsi": rsi_val},
                        )
                        return None
                    if direction == "SELL" and rsi_val <= 30:
                        self._reject(
                            pair_key,
                            pair,
                            "RSI_EXTREME",
                            f"SELL RSI {rsi_val:.1f} <= 30",
                            direction=direction,
                            hour_utc=eval_ts.hour,
                            details={"rsi": rsi_val},
                        )
                        return None

            # ── Confidence ────────────────────────────────────────────────────
            confidence = self._score_confidence(er_now, abs(kama_delta_pips), cross_er_min)
            if confidence < cfg.min_confidence:
                self._reject(
                    pair_key,
                    pair,
                    "CONFIDENCE",
                    f"confidence {confidence:.3f} < {cfg.min_confidence:.3f}",
                    direction=direction,
                    hour_utc=eval_ts.hour,
                    details={"confidence": confidence, "min_confidence": cfg.min_confidence},
                )
                return None
            confidence = min(confidence, cfg.max_confidence)

            # ── Entry / SL / TP ───────────────────────────────────────────────
            entry_price = close_now
            if direction == "BUY":
                stop_loss = entry_price - sl_pips * _PIP
                take_profit = entry_price + tp_pips * _PIP
            else:
                stop_loss = entry_price + sl_pips * _PIP
                take_profit = entry_price - tp_pips * _PIP

            # ── Update cooldown ───────────────────────────────────────────────
            self._cooldowns[pair_key] = eval_ts  # type: ignore[assignment]

            signal = {
                "strategy": "KAMA_V2",
                "signal": direction,
                "signal_type": direction,
                "signal_timestamp": str(eval_ts),
                "entry_price": entry_price,
                "stop_loss": round(stop_loss, 5),
                "take_profit": round(take_profit, 5),
                "risk_pips": sl_pips,
                "reward_pips": tp_pips,
                "confidence": round(confidence, 3),
                "confidence_score": round(confidence, 3),
                "timeframe": "5m",
                "pair": pair or pair_key,
                "epic": pair_key,
                "monitor_only": monitor_only,
                # Research was validated with fixed SL/TP — opt out of scalp trailing
                "is_scalp_trade": False,
                # Metadata for analysis
                "er": round(er_now, 4),
                "kama_delta_pips": round(kama_delta_pips, 2),
                "candle_time": str(eval_ts),
            }

            self.logger.info(
                "[KAMA_V2] %s: %s signal | ER=%.3f kama_slope=%.2f pips "
                "entry=%.5f SL=%.5f TP=%.5f conf=%.2f%s",
                pair_key, direction, er_now, kama_delta_pips,
                entry_price, stop_loss, take_profit, confidence,
                " [MONITOR]" if monitor_only else "",
            )

            return signal

        except Exception as exc:
            self.logger.error("[KAMA_V2] %s: detect_signal error: %s", pair or epic or "?", exc)
            return None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _score_confidence(self, er: float, abs_slope_pips: float, cross_er_min: float = 0.35) -> float:
        """Score 0.60–0.80: base + ER bonus (max +0.15) + slope bonus (max +0.05)."""
        base = self.config.base_confidence
        er_bonus = min(0.15, (er - cross_er_min) * 0.30)
        slope_bonus = min(0.05, abs_slope_pips * 0.02)
        return round(max(0.0, min(1.0, base + er_bonus + slope_bonus)), 4)

    def _reject(
        self,
        epic: str,
        pair: Optional[str],
        stage: str,
        reason: str,
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log and persist one strategy-level rejection."""
        pair_name = pair or epic
        self.logger.info("[KAMA_V2:REJECT] %s %s%s: %s", epic, stage, f" {direction}" if direction else "", reason)
        if self._rej_mgr is None:
            return
        try:
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair_name,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=self._current_timestamp,
                details=details,
            )
        except Exception as exc:
            self.logger.debug("[KAMA_V2] failed to track rejection for %s: %s", epic, exc)
