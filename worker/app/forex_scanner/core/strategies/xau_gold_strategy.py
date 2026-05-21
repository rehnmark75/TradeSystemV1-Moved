"""
XAU_GOLD Strategy — Gold-optimized multi-timeframe SMC-style strategy.

Design:
    Tier 1 (HTF bias, default 4H):   EMA(50/200) slope & order
    Tier 2 (trigger, default 1H):    Swing break / BOS + MACD alignment
    Tier 3 (entry, default 15m):     Pullback to fib zone (OB/FVG optional)

Gated by:
    - Regime: ADX trending (>=25), ATR percentile not in expansion zone
    - Session: London open + NY session (gold's liquidity windows)
    - RSI neutrality (40-60 band to avoid overextension)

Confidence: additive weighted sum on top of base_confidence, capped by
max_confidence (memory learning: very-high-confidence signals are
inversely predictive).

All pips reported in XAU pip units (pip size 0.1 = $0.10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .strategy_registry import StrategyInterface, register_strategy
except ImportError:  # pragma: no cover
    from forex_scanner.core.strategies.strategy_registry import (
        StrategyInterface,
        register_strategy,
    )

try:
    from ..alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]

try:
    from .helpers.smc_fair_value_gaps import FVGType, SMCFairValueGaps
except ImportError:  # pragma: no cover
    from forex_scanner.core.strategies.helpers.smc_fair_value_gaps import (
        FVGType,
        SMCFairValueGaps,
    )

try:
    from services.xau_gold_config_service import (
        XAUGoldConfig,
        XAUGoldConfigService,
        get_xau_gold_config,
        apply_config_overrides,
    )
except ImportError:  # pragma: no cover
    from forex_scanner.services.xau_gold_config_service import (
        XAUGoldConfig,
        XAUGoldConfigService,
        get_xau_gold_config,
        apply_config_overrides,
    )

try:
    from services.regime_classifier import compute_adx, get_adx_from_df
except ImportError:  # pragma: no cover
    from forex_scanner.services.regime_classifier import compute_adx, get_adx_from_df


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local indicator helpers (fallback when DataFetcher didn't pre-enrich)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)



def _macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd = _ema(s, fast) - _ema(s, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": hist})


def _ensure_col(df: pd.DataFrame, name: str, series: pd.Series) -> pd.DataFrame:
    if name not in df.columns:
        df = df.copy()
        df[name] = series
    return df


# ---------------------------------------------------------------------------
# Rejection record
# ---------------------------------------------------------------------------

@dataclass
class _Rejection:
    epic: str
    reason: str
    detail: str
    ts: datetime


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register_strategy("XAU_GOLD")
class XAUGoldStrategy(StrategyInterface):
    """Gold-optimized strategy (XAU/USD)."""

    def __init__(self, config=None, db_manager=None, logger=None, config_override=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        # Backtest parameter isolation: --override key=value pairs from bt.py
        self._config_override: Optional[Dict[str, Any]] = config_override
        # Prime config service (singleton)
        self._config_service = XAUGoldConfigService.get_instance()
        self.config: XAUGoldConfig = get_xau_gold_config()
        apply_config_overrides(self.config, self._config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._rejections: List[_Rejection] = []
        # BOS state persistence: keyed by epic, value = BOS dict with 'ts' key
        # Allows pullback detection in scans after BOS fires
        self._last_bos: Dict[str, Dict[str, Any]] = {}
        # BOS dedup: keyed by epic, value = set of (direction, from_price, to_price) tuples
        # that have already produced a signal. Prevents re-signaling the same structural break
        # on every scan until a genuinely new BOS bar appears.
        self._signaled_bos_keys: Dict[str, set] = {}
        self._adaptive_cache: Dict[Tuple[str, str, int], Tuple[datetime, Optional[Dict[str, Any]]]] = {}
        self._scan_count: int = 0
        self._rej_counts: Dict[str, int] = {}
        self._rej_last_log: datetime = datetime.now(timezone.utc)
        self._rej_log_interval_minutes: int = 15

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("XAU_GOLD", db_manager)
            except Exception:
                pass

        self.logger.info(f"XAU_GOLD strategy v{self.config.version} initialized")

    # ---- interface --------------------------------------------------------

    @property
    def strategy_name(self) -> str:
        return "XAU_GOLD"

    def get_required_timeframes(self) -> List[str]:
        c = self.config
        return [c.htf_timeframe, c.trigger_timeframe, c.entry_timeframe]

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    def flush_rejections(self) -> None:
        # Persist buffered rejections to DB, then clear per-scan list.
        if self._rej_mgr is not None:
            self._rej_mgr.flush()
        self._rejections.clear()

        if not self._rej_counts:
            return

        now = datetime.now(timezone.utc)
        elapsed = (now - self._rej_last_log).total_seconds() / 60.0
        if elapsed < self._rej_log_interval_minutes:
            return

        top = sorted(self._rej_counts.items(), key=lambda x: -x[1])[:6]
        total = sum(self._rej_counts.values())
        self.logger.info(
            f"[XAU_GOLD] Rejection rollup (last {int(elapsed)}min, {total} rejections): {dict(top)}"
        )
        self._rej_counts.clear()
        self._rej_last_log = now

    # ---- main dispatch ----------------------------------------------------

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        df_entry: Optional[pd.DataFrame] = None,
        df_5m: Optional[pd.DataFrame] = None,
        epic: str = "",
        pair: str = "",
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        # Refresh config on each cycle (cached by TTL inside service) and reapply
        # any backtest --override values on top of the refreshed config.
        self.config = get_xau_gold_config()
        apply_config_overrides(self.config, self._config_override)
        cfg = self.config

        if not cfg.is_pair_enabled(epic):
            return None

        # ema_slow_period can be 100-200; require enough warmup for EMA convergence
        # but stay below DataFetcher's ~99-bar practical cap on 4H
        if df_4h is None or len(df_4h) < min(80, cfg.ema_slow_period - 20):
            self._reject(epic, "insufficient_htf_data", f"bars={0 if df_4h is None else len(df_4h)}")
            return None
        if df_trigger is None or len(df_trigger) < 60:
            self._reject(epic, "insufficient_trigger_data", f"bars={0 if df_trigger is None else len(df_trigger)}")
            return None
        if df_entry is None or len(df_entry) < 30:
            self._reject(epic, "insufficient_entry_data", f"bars={0 if df_entry is None else len(df_entry)}")
            return None

        # Derive simulation time from the fastest available scan frame (works in
        # both live and backtest). XAU event gates are intraday/session-sensitive,
        # so using the 1H trigger timestamp can misclassify 5m/15m signals near
        # session boundaries.
        # DataFetcher returns start_time as a column (RangeIndex), not as the DataFrame index,
        # so we check the 'start_time' column first, then fall back to the DatetimeIndex,
        # then to wall-clock time as a last resort.
        ts_raw = None
        for time_df in (df_5m, df_entry, df_trigger):
            if time_df is None or len(time_df) == 0:
                continue
            if "start_time" in time_df.columns:
                ts_raw = time_df["start_time"].iloc[-1]
                break
            if isinstance(time_df.index, pd.DatetimeIndex):
                ts_raw = time_df.index[-1]
                break

        if ts_raw is not None:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            current_time: datetime = ts.to_pydatetime()
            hour_utc = current_time.hour
        else:
            current_time = datetime.now(timezone.utc)
            hour_utc = current_time.hour

        if not self._check_cooldown(epic, current_time):
            self._reject(epic, "cooldown", "")
            return None

        if not cfg.is_session_allowed(hour_utc):
            self._reject(epic, "session_blocked", f"hour_utc={hour_utc}")
            return None

        # Enrich timeframes with indicators if missing
        df_4h = self._enrich_htf(df_4h)
        df_trigger = self._enrich_trigger(df_trigger)

        if cfg.enable_event_playbooks and df_5m is not None and len(df_5m) >= 220:
            event_signal = self._detect_event_playbook_signal(
                df_5m=df_5m,
                df_4h=df_4h,
                df_trigger=df_trigger,
                epic=epic,
                pair=pair,
                current_time=current_time,
            )
            if event_signal is not None:
                return event_signal

        if not bool(getattr(cfg, "enable_strict_bos_pullback", True)):
            self._reject(epic, "strict_bos_pullback_disabled", "")
            return None

        # Tier 1: HTF bias
        bias = self._htf_bias(df_4h)
        if bias is None:
            self._reject(epic, "no_htf_bias", "EMA50 vs EMA200 unclear")
            return None

        # Regime gate (computed on 4H)
        regime, adx_val, atr_pct = self._regime(df_4h, df_trigger)
        # TRENDING-only design: block both ranging and the 20<=ADX<25 neutral band
        if cfg.block_ranging and regime in ("ranging", "neutral"):
            self._reject(epic, "regime_not_trending", f"adx={adx_val:.1f} regime={regime}")
            return None
        if cfg.block_expansion and regime == "expansion":
            self._reject(epic, "regime_expansion", f"atr_pct={atr_pct:.1f}")
            return None

        # DI alignment gate: block when momentum (DI) contradicts HTF bias
        if bool(getattr(cfg, "di_alignment_gate_enabled", True)):
            plus_di, minus_di = self._compute_di(df_trigger)
            if bias == "bearish" and plus_di > minus_di:
                self._reject(epic, "di_misaligned_sell", f"+DI={plus_di:.1f} > -DI={minus_di:.1f}")
                return None
            if bias == "bullish" and minus_di > plus_di:
                self._reject(epic, "di_misaligned_buy", f"-DI={minus_di:.1f} > +DI={plus_di:.1f}")
                return None

        # Tier 2: BOS on 1H aligned with bias — detect fresh OR use cached state
        fresh_bos = self._detect_bos(df_trigger, bias)
        if fresh_bos is not None:
            # Dedup: once a BOS has generated a signal, don't treat it as "fresh"
            # again on re-discovery within bos_search_bars. This prevents clusters of
            # duplicate signals from a single structural break across consecutive scans.
            bos_key = (
                fresh_bos["direction"],
                round(float(fresh_bos["from_price"]), 3),
                round(float(fresh_bos["to_price"]), 3),
            )
            if bos_key in self._signaled_bos_keys.get(epic, set()):
                fresh_bos = None  # already signaled this break — treat as cached only
            else:
                # Genuinely new break — store with simulation time
                fresh_bos["ts"] = current_time
                self._last_bos[epic] = fresh_bos

        # Use cached BOS if within expiry window (default 48h)
        bos_expiry_hours = cfg.bos_expiry_hours
        cached = self._last_bos.get(epic)
        if cached is not None:
            age_hours = (current_time - cached["ts"]).total_seconds() / 3600
            if age_hours > bos_expiry_hours or cached.get("direction") != bias:
                # Expired or bias flipped — discard cached BOS and signaled key history
                self._last_bos.pop(epic, None)
                self._signaled_bos_keys.pop(epic, None)
                cached = None

        bos = fresh_bos or cached
        if bos is None:
            self._reject(epic, "no_bos", f"bias={bias}")
            return None
        if cfg.macd_filter_enabled and not self._macd_aligned(df_trigger, bias):
            self._reject(epic, "macd_misaligned", f"bias={bias}")
            return None

        # Tier 3: 15m pullback into fib zone of the BOS leg
        entry = self._check_pullback_entry(df_entry, bos, bias)
        if entry is None:
            self._reject(epic, "no_pullback_entry", "not in fib zone yet")
            return None
        if cfg.require_ob_or_fvg and not self._entry_has_fvg_confluence(df_entry, entry, bias, epic):
            self._reject(epic, "missing_fvg_confluence", "entry lacks nearby active FVG")
            return None

        # RSI neutrality on trigger
        rsi_val = float(df_trigger["rsi_14"].iloc[-1])
        rsi_neutral = cfg.rsi_neutral_min <= rsi_val <= cfg.rsi_neutral_max

        # RSI directional floor: SELL into oversold / BUY into overbought are losing setups
        rsi_sell_floor = float(getattr(cfg, "rsi_sell_floor", 45.0))
        rsi_buy_ceiling = float(getattr(cfg, "rsi_buy_ceiling", 80.0))
        if bias == "bearish" and rsi_val < rsi_sell_floor:
            self._reject(epic, "rsi_sell_floor", f"rsi={rsi_val:.1f} < floor={rsi_sell_floor:.1f}")
            return None
        if bias == "bullish" and rsi_val > rsi_buy_ceiling:
            self._reject(epic, "rsi_buy_ceiling", f"rsi={rsi_val:.1f} > ceiling={rsi_buy_ceiling:.1f}")
            return None

        # DXY confluence is not wired yet; weight is zeroed in DB config to keep
        # the confidence scale honest. Setting False has no effect while w_dxy_confluence=0.
        dxy_confluence = False

        # Confidence
        confidence = self._confidence(
            bias_ok=True,
            bos_displacement_ok=bos.get("displacement_ok", False),
            entry_untested=entry.get("untested", True),
            rsi_neutral=rsi_neutral,
            dxy_confluence=dxy_confluence,
        )
        min_conf = cfg.get_pair_min_confidence(epic)
        max_conf = cfg.get_pair_max_confidence(epic)
        if confidence < min_conf:
            self._reject(epic, "low_confidence", f"{confidence:.3f} < {min_conf:.3f}")
            return None
        if confidence > max_conf:
            confidence = max_conf

        # SL / TP
        sl_pips, tp_pips = self._sl_tp(df_trigger, epic)
        rr = tp_pips / sl_pips if sl_pips > 0 else 0.0
        if rr < cfg.min_rr_ratio:
            self._reject(epic, "rr_too_low", f"rr={rr:.2f}")
            return None
        if tp_pips < cfg.min_tp_pips:
            self._reject(epic, "tp_too_small", f"tp={tp_pips:.1f}")
            return None

        entry_price = float(df_entry["close"].iloc[-1])
        direction = "BUY" if bias == "bullish" else "SELL"
        adaptive = self._adaptive_decision(direction, "strict_bos_pullback", current_time)
        if adaptive.get("blocked"):
            self._reject(
                epic,
                "adaptive_playbook_blocked",
                adaptive.get("reason", "strict_bos_pullback"),
            )
            return None
        if direction == "BUY" and not self._buy_session_allowed(
            current_time,
            adaptive=adaptive,
        ):
            self._reject(
                epic,
                "buy_session_blocked",
                f"hour_utc={current_time.hour}",
            )
            return None
        confidence = max(0.0, min(max_conf, confidence + float(adaptive.get("score_delta", 0.0))))

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair or "XAUUSD",
            "entry_price": entry_price,
            "price": entry_price,
            "stop_loss_pips": float(sl_pips),
            "take_profit_pips": float(tp_pips),
            # BacktestScanner reads risk_pips / reward_pips for trade simulation;
            # set them here so the evaluator uses the strategy's own SL/TP rather than
            # falling back to SMC-style trailing-config defaults (8+2=10 / 12 pips).
            "risk_pips": float(sl_pips),
            "reward_pips": float(tp_pips),
            # Order executor expects stop_distance / limit_distance in IG broker points.
            # For IG CFEGOLD, 1 broker point = 0.5 price units = 5 XAU pips.
            # Without this, order_executor falls back to default_stop_distance (~5 pips).
            "stop_distance": int(round(float(sl_pips) / 5.0)),
            "limit_distance": int(round(float(tp_pips) / 5.0)),
            "confidence_score": float(confidence),
            "confidence": float(confidence),
            "signal_timestamp": current_time.isoformat(),
            "timestamp": current_time,
            "version": self.config.version,
            "market_regime_detected": regime,
            "market_regime": regime,
            "adx_value": float(adx_val),
            "volatility_state": "expansion" if atr_pct >= cfg.atr_expansion_pct else "normal",
            "monitor_only": cfg.is_monitor_only(epic),
            "scalp_mode": False,
            "strategy_indicators": {
                # Timeframe hints for chart generator: 4h macro + 1h BOS trigger + 15m OB/FVG entry
                "tier1_ema": {"timeframe": "4h"},
                "tier2_swing": {"timeframe": "1h"},
                "tier3_entry": {"timeframe": "15m"},
                "bias": bias,
                "bos_from_price": bos.get("from_price"),
                "bos_to_price": bos.get("to_price"),
                "bos_displacement_atr": bos.get("displacement_atr"),
                "fib_depth": entry.get("fib_depth"),
                "entry_age_bars": entry.get("entry_age_bars"),
                "fvg_confluence": entry.get("fvg_confluence", False),
                "rsi_14": rsi_val,
                "atr_pct": float(atr_pct),
                "rr_ratio": float(rr),
                "adaptive_playbook": adaptive,
            },
        }

        # Record this BOS as signaled so re-discovery within bos_search_bars doesn't re-fire
        _signaled_bos_key = (
            bos["direction"],
            round(float(bos["from_price"]), 3),
            round(float(bos["to_price"]), 3),
        )
        self._signaled_bos_keys.setdefault(epic, set()).add(_signaled_bos_key)

        # Clear stored BOS after signal fires — require fresh BOS for next entry
        self._last_bos.pop(epic, None)
        self._set_cooldown(epic, current_time)
        self.logger.info(
            f"✅ [XAU_GOLD] {epic} {direction} @ {entry_price:.2f} "
            f"SL={sl_pips:.1f}p TP={tp_pips:.1f}p conf={confidence:.2f} regime={regime}"
        )

        # LPF gate — strategy-side opt-in (LPF_ENABLED = True)
        if getattr(self, 'LPF_ENABLED', True) and bool(getattr(self.config, "lpf_enabled", True)):
            try:
                try:
                    from .lpf_gate import apply_lpf_gate
                except ImportError:
                    from forex_scanner.core.strategies.lpf_gate import apply_lpf_gate
                signal = apply_lpf_gate(signal, self.logger, backtest_timestamp=current_time)
            except Exception as _lpf_exc:
                self.logger.warning("LPF gate error (letting signal through): %s", _lpf_exc)
        return signal

    # ---- tier helpers -----------------------------------------------------

    def _enrich_htf(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df = df.copy()
        df = _ensure_col(df, f"ema_{c.ema_fast_period}", _ema(df["close"], c.ema_fast_period))
        df = _ensure_col(df, f"ema_{c.ema_slow_period}", _ema(df["close"], c.ema_slow_period))
        df = _ensure_col(df, f"atr_{c.atr_period}", _atr(df, c.atr_period))
        df = _ensure_col(df, f"adx_{c.adx_period}", compute_adx(df, c.adx_period))
        return df

    def _enrich_trigger(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df = df.copy()
        df = _ensure_col(df, f"atr_{c.atr_period}", _atr(df, c.atr_period))
        df = _ensure_col(df, "rsi_14", _rsi(df["close"], c.rsi_period))
        if "macd_hist" not in df.columns:
            m = _macd(df["close"])
            df["macd"] = m["macd"]
            df["macd_signal"] = m["signal"]
            df["macd_hist"] = m["hist"]
        return df

    def _compute_di(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Return (plus_di, minus_di) scalars for the most recent bar using Wilder smoothing."""
        period = self.config.adx_period
        if len(df) < period + 1:
            return 0.0, 0.0
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
        )
        alpha = 1.0 / period
        tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        denom = float(tr_s.iloc[-1])
        if denom <= 0:
            return 0.0, 0.0
        return 100.0 * float(plus_s.iloc[-1]) / denom, 100.0 * float(minus_s.iloc[-1]) / denom

    def _enrich_event_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = _ensure_col(df, "ema_21", _ema(df["close"], 21))
        df = _ensure_col(df, "ema_50", _ema(df["close"], 50))
        df = _ensure_col(df, "ema_200", _ema(df["close"], 200))
        df = _ensure_col(df, "atr_14", _atr(df, 14))
        df = _ensure_col(df, "rsi_14", _rsi(df["close"], 14))
        body = (df["close"] - df["open"]).abs()
        df["body_atr"] = body / df["atr_14"].replace(0, np.nan)
        return df

    def _htf_bias(self, df: pd.DataFrame) -> Optional[str]:
        c = self.config
        fast = df[f"ema_{c.ema_fast_period}"].iloc[-1]
        slow = df[f"ema_{c.ema_slow_period}"].iloc[-1]
        price = float(df["close"].iloc[-1])
        # Require EMA ordering with a minimum gap (0.1% of price) to avoid noise near crossover
        min_gap = price * 0.001
        if fast > slow and (fast - slow) > min_gap:
            return "bullish"
        if fast < slow and (slow - fast) > min_gap:
            return "bearish"
        return None

    def _detect_event_playbook_signal(
        self,
        df_5m: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_trigger: pd.DataFrame,
        epic: str,
        pair: str,
        current_time: datetime,
    ) -> Optional[Dict[str, Any]]:
        """5m event-playbook layer for XAU.

        The original XAU_GOLD setup is a narrow 4H/1H/15m continuation pattern.
        This layer lets the 5-minute scanner surface more gold-specific events,
        then scores HTF context and path-risk instead of hard-blocking them.
        """
        cfg = self.config
        df = self._enrich_event_frame(df_5m)
        if len(df) < 220:
            return None

        i = len(df) - 1
        row = df.iloc[-1]
        prev = df.iloc[-2]
        pip = cfg.get_pip_size(epic)
        if pip <= 0:
            return None

        range_n = min(int(cfg.event_range_lookback_bars), len(df) - 2)
        micro_n = min(int(cfg.event_micro_break_lookback_bars), len(df) - 2)
        if range_n < 6 or micro_n < 4:
            return None

        prior_range = df.iloc[-range_n - 1 : -1]
        prior_micro = df.iloc[-micro_n - 1 : -1]
        range_high = float(prior_range["high"].max())
        range_low = float(prior_range["low"].min())
        micro_high = float(prior_micro["high"].max())
        micro_low = float(prior_micro["low"].min())
        recent_range_pips = (range_high - range_low) / pip
        body_atr = float(row.get("body_atr") or 0.0)
        rsi_val = float(row.get("rsi_14") or 50.0)
        close = float(row["close"])

        candidates: List[Dict[str, Any]] = []

        if close > range_high and body_atr >= cfg.event_min_body_atr:
            candidates.append({
                "setup": "range_break_2h",
                "direction": "BUY",
                "base": 0.56,
                "trigger_level": range_high,
                "event_strength": body_atr,
            })
        if close < range_low and body_atr >= cfg.event_min_body_atr:
            candidates.append({
                "setup": "range_break_2h",
                "direction": "SELL",
                "base": 0.56,
                "trigger_level": range_low,
                "event_strength": body_atr,
            })

        sweep_min = cfg.event_sweep_min_pips * pip
        candle_range = max(float(row["high"] - row["low"]), 1e-9)
        upper_reject = float(row["high"] - row["close"]) / candle_range
        lower_reject = float(row["close"] - row["low"]) / candle_range
        if row["high"] > range_high + sweep_min and close < range_high and upper_reject >= 0.45:
            candidates.append({
                "setup": "sweep_reversal_2h",
                "direction": "SELL",
                "base": 0.58,
                "trigger_level": range_high,
                "event_strength": upper_reject,
            })
        if row["low"] < range_low - sweep_min and close > range_low and lower_reject >= 0.45:
            candidates.append({
                "setup": "sweep_reversal_2h",
                "direction": "BUY",
                "base": 0.58,
                "trigger_level": range_low,
                "event_strength": lower_reject,
            })

        bull_stack = row["ema_21"] > row["ema_50"] > row["ema_200"]
        bear_stack = row["ema_21"] < row["ema_50"] < row["ema_200"]
        if bull_stack and prev["low"] <= prev["ema_21"] and close > row["ema_21"] and rsi_val < 70:
            candidates.append({
                "setup": "ema21_pullback_5m",
                "direction": "BUY",
                "base": 0.55,
                "trigger_level": float(row["ema_21"]),
                "event_strength": (close - float(row["ema_21"])) / pip,
            })
        if bear_stack and prev["high"] >= prev["ema_21"] and close < row["ema_21"] and rsi_val > 30:
            candidates.append({
                "setup": "ema21_pullback_5m",
                "direction": "SELL",
                "base": 0.55,
                "trigger_level": float(row["ema_21"]),
                "event_strength": (float(row["ema_21"]) - close) / pip,
            })

        if bull_stack and row["close"] > row["open"] and body_atr >= cfg.event_displacement_body_atr and close > micro_high:
            candidates.append({
                "setup": "displacement_5m",
                "direction": "BUY",
                "base": 0.57,
                "trigger_level": micro_high,
                "event_strength": body_atr,
            })
        if bear_stack and row["close"] < row["open"] and body_atr >= cfg.event_displacement_body_atr and close < micro_low:
            candidates.append({
                "setup": "displacement_5m",
                "direction": "SELL",
                "base": 0.57,
                "trigger_level": micro_low,
                "event_strength": body_atr,
            })

        if not candidates:
            return None

        htf_bias = self._htf_bias(df_4h)
        regime, adx_val, atr_pct = self._regime(df_4h, df_trigger)
        plus_di, minus_di = self._compute_di(df_trigger)
        ema_distance_pips = abs(close - float(row["ema_21"])) / pip

        scored = []
        filtered_candidates = 0
        for cand in candidates:
            direction = cand["direction"]
            adaptive = self._adaptive_decision(direction, cand["setup"], current_time)
            if adaptive.get("blocked"):
                filtered_candidates += 1
                continue
            if direction == "BUY" and not self._buy_session_allowed(
                current_time,
                adaptive=adaptive,
            ):
                filtered_candidates += 1
                continue

            # range_break_2h in Asian session (23-06 UTC) while ranging is low-edge noise
            hour_utc = current_time.hour
            is_asian = hour_utc >= 23 or hour_utc <= 5
            if cand["setup"] == "range_break_2h" and regime == "ranging" and is_asian:
                self._reject(
                    epic,
                    "range_break_asian_ranging",
                    f"hour_utc={hour_utc} regime={regime}",
                )
                filtered_candidates += 1
                continue

            # DI alignment gate: block when momentum (DI) contradicts direction
            if bool(getattr(cfg, "di_alignment_gate_enabled", True)):
                if direction == "SELL" and plus_di > minus_di:
                    self._reject(epic, "di_misaligned_sell", f"+DI={plus_di:.1f} > -DI={minus_di:.1f}")
                    filtered_candidates += 1
                    continue
                if direction == "BUY" and minus_di > plus_di:
                    self._reject(epic, "di_misaligned_buy", f"-DI={minus_di:.1f} > +DI={plus_di:.1f}")
                    filtered_candidates += 1
                    continue

            # RSI directional floor: SELL into oversold is a losing setup for gold
            rsi_sell_floor = float(getattr(cfg, "rsi_sell_floor", 45.0))
            if direction == "SELL" and rsi_val < rsi_sell_floor:
                self._reject(epic, "rsi_sell_floor", f"rsi={rsi_val:.1f} < floor={rsi_sell_floor:.1f}")
                filtered_candidates += 1
                continue
            rsi_buy_ceiling = float(getattr(cfg, "rsi_buy_ceiling", 80.0))
            if direction == "BUY" and rsi_val > rsi_buy_ceiling:
                self._reject(epic, "rsi_buy_ceiling", f"rsi={rsi_val:.1f} > ceiling={rsi_buy_ceiling:.1f}")
                filtered_candidates += 1
                continue

            score = float(cand["base"])
            aligned = (direction == "BUY" and htf_bias == "bullish") or (
                direction == "SELL" and htf_bias == "bearish"
            )
            if aligned:
                score += 0.08
            elif htf_bias is not None and cand["setup"] != "sweep_reversal_2h":
                score -= 0.08

            if regime == "trending":
                score += 0.04
            elif cand["setup"] in {"sweep_reversal_2h", "range_break_2h"} and regime == "neutral":
                score += 0.02

            if recent_range_pips <= cfg.event_max_recent_range_pips:
                score += 0.05
            else:
                score -= min(0.10, (recent_range_pips - cfg.event_max_recent_range_pips) / 1000.0)

            if ema_distance_pips <= cfg.event_max_ema_distance_pips:
                score += 0.03
            else:
                score -= min(0.10, (ema_distance_pips - cfg.event_max_ema_distance_pips) / 1000.0)

            if direction == "BUY" and rsi_val > 76:
                score -= 0.07
            if direction == "SELL" and rsi_val < 24:
                score -= 0.07

            hour_utc = current_time.hour
            if 5 <= hour_utc <= 16:
                score += 0.02

            score += float(adaptive.get("score_delta", 0.0))

            profile = self._event_profile_decision(
                direction=direction,
                setup=cand["setup"],
                current_time=current_time,
                score=score,
                metrics={
                    "rsi_14": rsi_val,
                    "body_atr": body_atr,
                    "recent_range_pips": recent_range_pips,
                    "ema21_distance_pips": ema_distance_pips,
                    "event_strength": float(cand.get("event_strength") or 0.0),
                    "atr_pct": float(atr_pct),
                },
            )
            if not profile.get("allowed", True):
                filtered_candidates += 1
                continue
            score += float(profile.get("score_delta", 0.0))

            cand = {
                **cand,
                "score": min(max(score, 0.0), cfg.max_confidence),
                "adaptive": adaptive,
                "profile": profile,
            }
            scored.append(cand)

        if not scored:
            self._reject(
                epic,
                "event_adaptive_filtered",
                f"filtered={filtered_candidates} candidates={len(candidates)}",
            )
            return None

        best = max(scored, key=lambda c: c["score"])
        profile = best.get("profile") or {}
        min_confidence = float(profile.get("min_confidence", cfg.event_min_confidence))
        if best["score"] < min_confidence:
            self._reject(
                epic,
                "event_low_confidence",
                f"{best['setup']} {best['score']:.3f} < {min_confidence:.3f}",
            )
            return None

        sl_pips, tp_pips = self._sl_tp(df_trigger, epic)
        direction = best["direction"]
        if direction == "BUY" and not self._buy_session_allowed(current_time):
            self._reject(
                epic,
                "buy_session_blocked",
                f"{best['setup']} hour_utc={current_time.hour}",
            )
            return None

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair or "XAUUSD",
            "entry_price": close,
            "price": close,
            "stop_loss_pips": float(sl_pips),
            "take_profit_pips": float(tp_pips),
            "risk_pips": float(sl_pips),
            "reward_pips": float(tp_pips),
            "stop_distance": int(round(float(sl_pips) / 5.0)),
            "limit_distance": int(round(float(tp_pips) / 5.0)),
            "confidence_score": float(best["score"]),
            "confidence": float(best["score"]),
            "signal_timestamp": current_time.isoformat(),
            "timestamp": current_time,
            "version": self.config.version,
            "market_regime_detected": regime,
            "market_regime": regime,
            "adx_value": float(adx_val),
            "volatility_state": "expansion" if atr_pct >= cfg.atr_expansion_pct else "normal",
            "monitor_only": cfg.is_monitor_only(epic),
            "scalp_mode": False,
            "strategy_indicators": {
                # Timeframe hints for chart generator: 4h macro + 1h BOS trigger + 15m OB/FVG entry
                "tier1_ema": {"timeframe": "4h"},
                "tier2_swing": {"timeframe": "1h"},
                "tier3_entry": {"timeframe": "15m"},
                "xau_playbook": best["setup"],
                "xau_event_layer": True,
                "htf_bias": htf_bias,
                "htf_aligned": (direction == "BUY" and htf_bias == "bullish") or (
                    direction == "SELL" and htf_bias == "bearish"
                ),
                "trigger_level": best.get("trigger_level"),
                "event_strength": best.get("event_strength"),
                "event_score": best["score"],
                "body_atr": body_atr,
                "rsi_14": rsi_val,
                "recent_range_pips": recent_range_pips,
                "ema21_distance_pips": ema_distance_pips,
                "atr_pct": float(atr_pct),
                "rr_ratio": float(tp_pips / sl_pips) if sl_pips else 0.0,
                "adaptive_playbook": best.get("adaptive"),
                "event_profile": profile,
            },
        }

        cooldown_minutes = int(profile.get("cooldown_minutes", cfg.event_cooldown_minutes))
        self._set_cooldown_minutes(epic, current_time, cooldown_minutes)
        self.logger.info(
            f"✅ [XAU_GOLD:event] {epic} {direction} {best['setup']} @ {close:.2f} "
            f"SL={sl_pips:.1f}p TP={tp_pips:.1f}p conf={best['score']:.2f} regime={regime}"
        )

        if getattr(self, 'LPF_ENABLED', True) and bool(getattr(self.config, "lpf_enabled", True)):
            try:
                try:
                    from .lpf_gate import apply_lpf_gate
                except ImportError:
                    from forex_scanner.core.strategies.lpf_gate import apply_lpf_gate
                signal = apply_lpf_gate(signal, self.logger, backtest_timestamp=current_time)
            except Exception as _lpf_exc:
                self.logger.warning("LPF gate error (letting signal through): %s", _lpf_exc)
        return signal

    def _event_profile_decision(
        self,
        direction: str,
        setup: str,
        current_time: datetime,
        score: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        metrics = metrics or {}
        result: Dict[str, Any] = {
            "enabled": bool(cfg.event_dynamic_profiles_enabled),
            "allowed": True,
            "matched_rule": None,
            "min_confidence": float(cfg.event_min_confidence),
            "cooldown_minutes": int(cfg.event_cooldown_minutes),
            "score_delta": 0.0,
        }
        if not cfg.event_dynamic_profiles_enabled:
            return result

        rules = cfg.event_profile_rules or []
        if not isinstance(rules, list):
            result["allowed"] = False
            result["reason"] = "invalid_profile_rules"
            return result

        hour = current_time.hour
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            setups = rule.get("setups", rule.get("setup", "*"))
            directions = rule.get("directions", rule.get("direction", "*"))
            hours = rule.get("hours")

            if not self._profile_value_matches(setup, setups):
                continue
            if not self._profile_value_matches(direction, directions):
                continue
            if hours is not None and not self._profile_hour_matches(hour, hours):
                continue

            metric_failure = self._profile_metric_failure(rule, metrics)
            if metric_failure is not None:
                result["matched_rule"] = idx
                result["allowed"] = False
                result["reason"] = metric_failure
                return result

            result["matched_rule"] = idx
            result["allowed"] = bool(rule.get("enabled", True))
            if "min_confidence" in rule:
                result["min_confidence"] = float(rule["min_confidence"])
            if "cooldown_minutes" in rule:
                result["cooldown_minutes"] = int(rule["cooldown_minutes"])
            if "score_delta" in rule:
                result["score_delta"] = float(rule["score_delta"])
            if score + float(result["score_delta"]) < float(result["min_confidence"]):
                result["reason"] = "below_profile_min_confidence"
            else:
                result["reason"] = "profile_matched"
            return result

        result["allowed"] = False
        result["reason"] = "no_profile_match"
        return result

    @staticmethod
    def _profile_value_matches(value: str, allowed: Any) -> bool:
        if allowed in (None, "*"):
            return True
        if isinstance(allowed, str):
            return value.upper() == allowed.upper()
        if isinstance(allowed, (list, tuple, set)):
            return any(XAUGoldStrategy._profile_value_matches(value, item) for item in allowed)
        return False

    @staticmethod
    def _profile_hour_matches(hour: int, hours: Any) -> bool:
        if hours in (None, "*"):
            return True
        if isinstance(hours, int):
            return hour == hours
        if isinstance(hours, str):
            try:
                return hour == int(hours)
            except ValueError:
                return False
        if isinstance(hours, (list, tuple)):
            if len(hours) == 2 and all(isinstance(x, (int, float, str)) for x in hours):
                try:
                    start = int(hours[0])
                    end = int(hours[1])
                except ValueError:
                    return False
                if start <= end:
                    return start <= hour <= end
                return hour >= start or hour <= end
            return any(XAUGoldStrategy._profile_hour_matches(hour, item) for item in hours)
        return False

    @staticmethod
    def _profile_metric_failure(rule: Dict[str, Any], metrics: Dict[str, float]) -> Optional[str]:
        aliases = {
            "rsi": "rsi_14",
            "range": "recent_range_pips",
            "recent_range": "recent_range_pips",
            "ema_distance": "ema21_distance_pips",
            "ema_dist": "ema21_distance_pips",
        }
        for key, raw_value in rule.items():
            if key.startswith("min_"):
                metric_name = aliases.get(key[4:], key[4:])
                if metric_name in {"confidence", "conf"}:
                    continue
                metric = metrics.get(metric_name)
                if metric is not None and metric < float(raw_value):
                    return f"{metric_name}_below_min"
            elif key.startswith("max_"):
                metric_name = aliases.get(key[4:], key[4:])
                if metric_name in {"confidence", "conf"}:
                    continue
                metric = metrics.get(metric_name)
                if metric is not None and metric > float(raw_value):
                    return f"{metric_name}_above_max"
        return None

    def _regime(
        self, df_htf: pd.DataFrame, df_trigger: pd.DataFrame
    ) -> Tuple[str, float, float]:
        c = self.config
        adx_val = get_adx_from_df(df_trigger, c.adx_period) or 0.0
        atr_series = df_htf[f"atr_{c.atr_period}"]
        lookback = min(c.atr_pct_lookback_bars, len(atr_series))
        recent = atr_series.iloc[-lookback:]
        current_atr = float(recent.iloc[-1])
        # Percentile rank (0-100) of current ATR within recent window
        atr_pct = float((recent <= current_atr).sum() / len(recent) * 100.0)

        if atr_pct >= c.atr_expansion_pct:
            return "expansion", adx_val, atr_pct
        if adx_val >= c.adx_trending_threshold:
            return "trending", adx_val, atr_pct
        if adx_val < c.adx_ranging_threshold:
            return "ranging", adx_val, atr_pct
        return "neutral", adx_val, atr_pct

    def _detect_bos(self, df: pd.DataFrame, bias: str) -> Optional[Dict[str, Any]]:
        """Detect the most recent swing break aligned with bias within the last bos_search_bars.

        Instead of requiring the *current* bar to be the BOS bar, we scan the recent history
        for the most recent bar that broke above/below the preceding swing_lookback bars.
        This prevents the strategy from only firing during the exact BOS candle.
        """
        c = self.config
        lookback = min(c.swing_lookback, len(df) - 2)
        if lookback < 5:
            return None
        atr = float(df[f"atr_{c.atr_period}"].iloc[-1])
        if atr <= 0:
            return None

        # How many recent bars to search for a BOS (default 48 = 48h on 1H)
        bos_search_bars = int(getattr(c, "bos_search_bars", 48))
        search_n = min(bos_search_bars, len(df) - lookback - 2)
        if search_n < 1:
            return None

        # Scan from most recent → oldest; return first valid BOS found
        # offset=1 checks the most recent bar (df.iloc[-1]) against the preceding swing
        n = len(df)
        for offset in range(1, search_n + 1):
            bar_idx = n - offset          # the candidate BOS bar position
            if bar_idx <= lookback:
                break
            bar = df.iloc[bar_idx]        # the candidate BOS bar itself
            window = df.iloc[bar_idx - lookback : bar_idx]  # preceding bars for swing level

            if bias == "bullish":
                swing_high = float(window["high"].max())
                if bar["close"] > swing_high:
                    displacement = float(bar["close"] - swing_high)
                    return {
                        "direction": "bullish",
                        "from_price": swing_high,
                        "to_price": float(bar["close"]),
                        "displacement_atr": displacement / atr,
                        "displacement_ok": displacement / atr >= c.bos_displacement_atr_mult,
                    }
            else:  # bearish
                swing_low = float(window["low"].min())
                if bar["close"] < swing_low:
                    displacement = float(swing_low - bar["close"])
                    return {
                        "direction": "bearish",
                        "from_price": swing_low,
                        "to_price": float(bar["close"]),
                        "displacement_atr": displacement / atr,
                        "displacement_ok": displacement / atr >= c.bos_displacement_atr_mult,
                    }

        return None

    def _macd_aligned(self, df: pd.DataFrame, bias: str) -> bool:
        hist = float(df["macd_hist"].iloc[-1])
        return (bias == "bullish" and hist > 0) or (bias == "bearish" and hist < 0)

    def _check_pullback_entry(
        self, df_entry: pd.DataFrame, bos: Dict[str, Any], bias: str
    ) -> Optional[Dict[str, Any]]:
        """Look for a pullback on 15m into the fib zone of the BOS impulse.

        Scans the most recent entry_check_bars for any bar
        whose close is in the fib zone. This allows the strategy to fire when price
        RECENTLY touched the pullback zone, but still keeps the setup fresh.
        """
        c = self.config
        from_p = float(bos["from_price"])
        to_p = float(bos["to_price"])
        leg = to_p - from_p  # positive for bullish, negative for bearish
        if leg == 0:
            return None

        # Zone boundaries (price range)
        if bias == "bullish":
            zone_low = to_p - leg * c.fib_pullback_max   # deeper pullback
            zone_high = to_p - leg * c.fib_pullback_min  # shallow pullback
        else:  # bearish: leg < 0
            zone_low = to_p + (-leg) * c.fib_pullback_min
            zone_high = to_p + (-leg) * c.fib_pullback_max
        lo, hi = sorted([zone_low, zone_high])

        # How many recent 15m bars to scan for a valid entry touch
        check_bars = int(getattr(c, "entry_check_bars", 48))
        scan_df = df_entry.iloc[-check_bars:] if len(df_entry) >= check_bars else df_entry

        # Scan from most recent → oldest; return first (most recent) hit
        for i in range(len(scan_df) - 1, -1, -1):
            bar = scan_df.iloc[i]
            price = float(bar["close"])
            # Require close inside the fib zone — wick-only touches are excluded
            # to avoid entering on false breakdowns that immediately reverse.
            if not (lo <= price <= hi):
                continue
            # Depth is bounded within [fib_min, fib_max] because close is inside the zone
            depth = (to_p - price) / leg if bias == "bullish" else (price - to_p) / (-leg)
            # Untested: no bar prior to this one has been in the zone
            prior = scan_df.iloc[max(0, i - 10) : i]
            untested = not ((prior["low"] <= hi) & (prior["high"] >= lo)).any() if len(prior) > 0 else True
            entry_age_bars = len(scan_df) - 1 - i
            return {
                "fib_depth": float(depth),
                "untested": bool(untested),
                "entry_index": int(i),
                "entry_age_bars": int(entry_age_bars),
                    "entry_price": float(price),
                    "zone_low": float(lo),
                    "zone_high": float(hi),
                }

        return None

    def _entry_has_fvg_confluence(
        self,
        df_entry: pd.DataFrame,
        entry: Dict[str, Any],
        bias: str,
        epic: str,
    ) -> bool:
        """Require the selected entry touch to occur inside or very near an active FVG."""
        entry_idx_in_scan = entry.get("entry_index")
        if entry_idx_in_scan is None:
            return False

        # Reconstruct the absolute position of the entry bar in df_entry. The entry_index
        # is returned as a position within the last `entry_check_bars` slice used by
        # _check_pullback_entry, not within df_entry itself.
        c = self.config
        scan_len = min(len(df_entry), int(getattr(c, "entry_check_bars", 48)))
        abs_entry_idx = len(df_entry) - scan_len + int(entry_idx_in_scan)
        if abs_entry_idx < 2 or abs_entry_idx >= len(df_entry):
            return False

        # Detect FVGs on a window of up to 96 bars ending at the entry bar
        fvg_lookback = min(abs_entry_idx + 1, 96)
        upto_entry = df_entry.iloc[abs_entry_idx + 1 - fvg_lookback : abs_entry_idx + 1].copy()
        if len(upto_entry) < 3:
            return False

        detector = SMCFairValueGaps(self.logger)
        pip_value = self.config.get_pip_size(epic)
        detector.detect_fair_value_gaps(
            upto_entry,
            {
                "epic": epic,
                "pip_value": pip_value,
                "fvg_min_size": 8,
                "fvg_max_age": 24,
                "fvg_fill_threshold": 0.8,
                "max_distance_to_zone": 5,
            },
        )

        target_type = FVGType.BULLISH if bias == "bullish" else FVGType.BEARISH
        candle = upto_entry.iloc[-1]
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        for fvg in detector.fair_value_gaps:
            if fvg.gap_type != target_type:
                continue
            if fvg.status.name not in {"ACTIVE", "PARTIALLY_FILLED"}:
                continue
            if candle_high >= fvg.low_price and candle_low <= fvg.high_price:
                entry["fvg_confluence"] = True
                return True

        return False

    # ---- confidence / SL TP ----------------------------------------------

    def _confidence(
        self,
        bias_ok: bool,
        bos_displacement_ok: bool,
        entry_untested: bool,
        rsi_neutral: bool,
        dxy_confluence: bool,
    ) -> float:
        c = self.config
        score = c.base_confidence
        if bias_ok:
            score += c.w_htf_bias
        if bos_displacement_ok:
            score += c.w_bos_displacement
        if entry_untested:
            score += c.w_entry_pullback
        if dxy_confluence:
            score += c.w_dxy_confluence
        if rsi_neutral:
            score += c.w_rsi_neutral
        return min(score, c.max_confidence)

    def _sl_tp(self, df_trigger: pd.DataFrame, epic: str) -> Tuple[float, float]:
        c = self.config
        pip = c.get_pip_size(epic)

        fixed_sl = c.get_pair_fixed_stop_loss(epic)
        fixed_tp = c.get_pair_fixed_take_profit(epic)
        if fixed_sl is not None and fixed_tp is not None:
            return float(fixed_sl), float(fixed_tp)

        atr_price = float(df_trigger[f"atr_{c.atr_period}"].iloc[-1])
        atr_pips = atr_price / pip if pip > 0 else atr_price
        sl_pips = c.get_pair_sl_atr_mult(epic) * atr_pips
        sl_pips = max(c.min_stop_loss_pips, min(c.max_stop_loss_pips, sl_pips))
        tp_pips = sl_pips * c.get_pair_rr_ratio(epic)
        tp_pips = max(c.min_tp_pips, tp_pips)
        return float(sl_pips), float(tp_pips)

    # ---- cooldown ---------------------------------------------------------

    def _check_cooldown(self, epic: str, current_time: Optional[datetime] = None) -> bool:
        end = self._cooldowns.get(epic)
        if end is None:
            return True
        now = current_time or datetime.now(timezone.utc)
        if now >= end:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str, current_time: Optional[datetime] = None) -> None:
        minutes = self.config.get_pair_cooldown_minutes(epic)
        self._set_cooldown_minutes(epic, current_time, minutes)

    def _set_cooldown_minutes(
        self,
        epic: str,
        current_time: Optional[datetime],
        minutes: int,
    ) -> None:
        now = current_time or datetime.now(timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=minutes)

    def _adaptive_decision(
        self,
        direction: str,
        setup: str,
        current_time: datetime,
    ) -> Dict[str, Any]:
        cfg = self.config
        result: Dict[str, Any] = {
            "enabled": bool(cfg.adaptive_playbook_scoring_enabled),
            "direction": direction,
            "setup": setup,
            "sample": 0,
            "score_delta": 0.0,
            "blocked": False,
            "buy_override": False,
        }
        if not cfg.adaptive_playbook_scoring_enabled:
            return result

        stats = self._get_adaptive_stats(direction, setup, current_time)
        if not stats:
            result["reason"] = "no_history"
            return result

        result.update(stats)
        if int(stats.get("sample", 0)) < int(cfg.adaptive_min_trades):
            result["reason"] = "insufficient_sample"
            return result

        pf = float(stats.get("profit_factor") or 0.0)
        expectancy = float(stats.get("expectancy_pips") or 0.0)

        if pf <= cfg.adaptive_block_pf and expectancy <= cfg.adaptive_block_expectancy_pips:
            result["blocked"] = True
            result["reason"] = f"pf={pf:.2f} expectancy={expectancy:.1f}"
            return result

        if pf <= cfg.adaptive_penalty_pf or expectancy <= cfg.adaptive_penalty_expectancy_pips:
            result["score_delta"] = -float(cfg.adaptive_score_penalty)
            result["reason"] = f"penalty pf={pf:.2f} expectancy={expectancy:.1f}"
        elif pf >= cfg.adaptive_bonus_pf and expectancy >= cfg.adaptive_bonus_expectancy_pips:
            result["score_delta"] = float(cfg.adaptive_score_bonus)
            result["buy_override"] = True
            result["reason"] = f"bonus pf={pf:.2f} expectancy={expectancy:.1f}"
        else:
            result["reason"] = f"neutral pf={pf:.2f} expectancy={expectancy:.1f}"

        return result

    def _get_adaptive_stats(
        self,
        direction: str,
        setup: str,
        current_time: datetime,
    ) -> Optional[Dict[str, Any]]:
        if self.db_manager is None:
            return None

        direction_key = "BULL" if direction == "BUY" else "BEAR"
        bucket = int(current_time.timestamp() // 900)
        cache_key = (direction_key, setup, bucket)
        cached = self._adaptive_cache.get(cache_key)
        if cached is not None:
            return cached[1]

        start_time = current_time - timedelta(days=int(self.config.adaptive_lookback_days))
        params = {
            "direction": direction_key,
            "setup": setup,
            "start_time": start_time.replace(tzinfo=None),
            "end_time": current_time.replace(tzinfo=None),
        }

        query = """
            WITH outcome_rows AS (
                SELECT
                    signal_type,
                    COALESCE(NULLIF(indicator_values->>'xau_playbook', ''), 'strict_bos_pullback') AS setup,
                    signal_timestamp AS outcome_time,
                    trade_result,
                    COALESCE(pips_gained, 0) AS pips_gained
                FROM backtest_signals
                WHERE strategy_name = 'XAU_GOLD'
                  AND epic = 'CS.D.CFEGOLD.CEE.IP'
                  AND trade_result IN ('win', 'loss', 'breakeven')

                UNION ALL

                SELECT
                    CASE
                        WHEN UPPER(ah.signal_type) IN ('BUY', 'LONG', 'BULL') THEN 'BULL'
                        WHEN UPPER(ah.signal_type) IN ('SELL', 'SHORT', 'BEAR') THEN 'BEAR'
                        ELSE UPPER(ah.signal_type)
                    END AS signal_type,
                    COALESCE(NULLIF(ah.strategy_indicators::jsonb->>'xau_playbook', ''), 'strict_bos_pullback') AS setup,
                    COALESCE(t.closed_at, t.pnl_calculated_at, t.updated_at, t.timestamp) AS outcome_time,
                    CASE
                        WHEN COALESCE(t.pips_gained, 0) > 0 THEN 'win'
                        WHEN COALESCE(t.pips_gained, 0) < 0 THEN 'loss'
                        ELSE 'breakeven'
                    END AS trade_result,
                    COALESCE(t.pips_gained, 0) AS pips_gained
                FROM trade_log t
                JOIN alert_history ah ON ah.id = t.alert_id
                WHERE ah.strategy = 'XAU_GOLD'
                  AND ah.epic = 'CS.D.CFEGOLD.CEE.IP'
                  AND t.pips_gained IS NOT NULL
            )
            SELECT
                COUNT(*)::int AS sample,
                SUM(CASE WHEN trade_result = 'win' THEN 1 ELSE 0 END)::int AS wins,
                AVG(COALESCE(pips_gained, 0))::float AS expectancy_pips,
                COALESCE(SUM(CASE WHEN COALESCE(pips_gained, 0) > 0 THEN pips_gained ELSE 0 END), 0)::float AS gross_profit,
                ABS(COALESCE(SUM(CASE WHEN COALESCE(pips_gained, 0) < 0 THEN pips_gained ELSE 0 END), 0))::float AS gross_loss
            FROM outcome_rows
            WHERE signal_type = :direction
              AND setup = :setup
              AND outcome_time >= :start_time
              AND outcome_time < :end_time
        """

        try:
            df = self.db_manager.execute_query(query, params)
            if df.empty:
                stats = None
            else:
                row = df.iloc[0]
                sample = int(row.get("sample") or 0)
                gross_profit = float(row.get("gross_profit") or 0.0)
                gross_loss = float(row.get("gross_loss") or 0.0)
                if sample <= 0:
                    stats = None
                else:
                    stats = {
                        "sample": sample,
                        "wins": int(row.get("wins") or 0),
                        "win_rate": float(row.get("wins") or 0) / sample,
                        "expectancy_pips": float(row.get("expectancy_pips") or 0.0),
                        "profit_factor": 99.0 if gross_loss <= 0 and gross_profit > 0 else (
                            gross_profit / gross_loss if gross_loss > 0 else 0.0
                        ),
                    }
        except Exception as exc:
            self.logger.debug("XAU adaptive stats unavailable: %s", exc)
            stats = None

        if len(self._adaptive_cache) > 512:
            self._adaptive_cache.clear()
        self._adaptive_cache[cache_key] = (current_time, stats)
        return stats

    def _buy_session_allowed(
        self,
        current_time: datetime,
        adaptive: Optional[Dict[str, Any]] = None,
    ) -> bool:
        c = self.config
        if not c.buy_session_gate_enabled:
            return True
        if (
            c.buy_session_adaptive_override_enabled
            and adaptive
            and adaptive.get("buy_override")
        ):
            return True
        start = int(c.buy_session_start_hour_utc)
        end = int(c.buy_session_end_hour_utc)
        hour = current_time.hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    # ---- rejection --------------------------------------------------------

    def _reject(self, epic: str, reason: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc)
        self._rejections.append(_Rejection(epic=epic, reason=reason, detail=detail, ts=now))
        self._rej_counts[reason] = self._rej_counts.get(reason, 0) + 1
        self._scan_count += 1
        self.logger.debug(f"[XAU_GOLD] {epic} rejected: {reason} ({detail})")
        if self._rej_mgr is not None:
            self._rej_mgr.reject(
                stage=reason,
                reason=detail or reason,
                epic=epic,
                pair=epic,
                scan_timestamp=now,
                details={"detail": detail} if detail else None,
            )


def create_xau_gold_strategy(
    config=None, db_manager=None, logger=None, config_override=None
) -> XAUGoldStrategy:
    return XAUGoldStrategy(
        config=config,
        db_manager=db_manager,
        logger=logger,
        config_override=config_override,
    )
