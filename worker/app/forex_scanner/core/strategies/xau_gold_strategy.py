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


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


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
        self._scan_count: int = 0
        self._rej_counts: Dict[str, int] = {}
        self._rej_last_log: datetime = datetime.now(timezone.utc)
        self._rej_log_interval_minutes: int = 15
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
        # Clear the per-scan rejection list. Counters in self._rej_counts keep
        # accumulating across scans so the rollup below can report an aggregate.
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

        # Derive simulation time from trigger data (works in both live and backtest).
        # DataFetcher returns start_time as a column (RangeIndex), not as the DataFrame index,
        # so we check the 'start_time' column first, then fall back to the DatetimeIndex,
        # then to wall-clock time as a last resort.
        ts_raw = None
        if "start_time" in df_trigger.columns and len(df_trigger) > 0:
            ts_raw = df_trigger["start_time"].iloc[-1]
        elif isinstance(df_trigger.index, pd.DatetimeIndex) and len(df_trigger) > 0:
            ts_raw = df_trigger.index[-1]

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

        # Tier 2: BOS on 1H aligned with bias — detect fresh OR use cached state
        fresh_bos = self._detect_bos(df_trigger, bias)
        if fresh_bos is not None:
            # Store fresh BOS with simulation time for persistence across scans
            fresh_bos["ts"] = current_time
            self._last_bos[epic] = fresh_bos

        # Use cached BOS if within expiry window (default 48h)
        bos_expiry_hours = cfg.bos_expiry_hours
        cached = self._last_bos.get(epic)
        if cached is not None:
            age_hours = (current_time - cached["ts"]).total_seconds() / 3600
            if age_hours > bos_expiry_hours or cached.get("direction") != bias:
                # Expired or bias flipped — discard
                self._last_bos.pop(epic, None)
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

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair or "XAUUSD",
            "entry_price": entry_price,
            "stop_loss_pips": float(sl_pips),
            "take_profit_pips": float(tp_pips),
            # BacktestScanner reads risk_pips / reward_pips for trade simulation;
            # set them here so the evaluator uses the strategy's own SL/TP rather than
            # falling back to SMC-style trailing-config defaults (8+2=10 / 12 pips).
            "risk_pips": float(sl_pips),
            "reward_pips": float(tp_pips),
            "confidence_score": float(confidence),
            "confidence": float(confidence),
            "signal_timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp": datetime.now(timezone.utc),
            "version": self.config.version,
            "market_regime_detected": regime,
            "market_regime": regime,
            "adx_value": float(adx_val),
            "volatility_state": "expansion" if atr_pct >= cfg.atr_expansion_pct else "normal",
            "monitor_only": cfg.is_monitor_only(epic),
            "strategy_indicators": {
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
            },
        }

        # Clear stored BOS after signal fires — require fresh BOS for next entry
        self._last_bos.pop(epic, None)
        self._set_cooldown(epic, current_time)
        self.logger.info(
            f"✅ [XAU_GOLD] {epic} {direction} @ {entry_price:.2f} "
            f"SL={sl_pips:.1f}p TP={tp_pips:.1f}p conf={confidence:.2f} regime={regime}"
        )
        return signal

    # ---- tier helpers -----------------------------------------------------

    def _enrich_htf(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df = df.copy()
        df = _ensure_col(df, f"ema_{c.ema_fast_period}", _ema(df["close"], c.ema_fast_period))
        df = _ensure_col(df, f"ema_{c.ema_slow_period}", _ema(df["close"], c.ema_slow_period))
        df = _ensure_col(df, f"atr_{c.atr_period}", _atr(df, c.atr_period))
        df = _ensure_col(df, f"adx_{c.adx_period}", _adx(df, c.adx_period))
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

    def _regime(
        self, df_htf: pd.DataFrame, df_trigger: pd.DataFrame
    ) -> Tuple[str, float, float]:
        c = self.config
        adx_val = float(df_trigger.get(f"adx_{c.adx_period}", _adx(df_trigger, c.adx_period)).iloc[-1])
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
            # Also allow the bar's high/low to touch the zone (not just close)
            bar_lo = float(bar.get("low", price))
            bar_hi = float(bar.get("high", price))
            if bar_hi >= lo and bar_lo <= hi:
                # Compute depth from close
                depth = (to_p - price) / leg if bias == "bullish" else (price - to_p) / (-leg)
                depth = max(0.0, depth)  # clip to non-negative
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
        now = current_time or datetime.now(timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=minutes)

    # ---- rejection --------------------------------------------------------

    def _reject(self, epic: str, reason: str, detail: str = "") -> None:
        self._rejections.append(
            _Rejection(epic=epic, reason=reason, detail=detail, ts=datetime.now(timezone.utc))
        )
        self._rej_counts[reason] = self._rej_counts.get(reason, 0) + 1
        self._scan_count += 1
        self.logger.debug(f"[XAU_GOLD] {epic} rejected: {reason} ({detail})")


def create_xau_gold_strategy(
    config=None, db_manager=None, logger=None, config_override=None
) -> XAUGoldStrategy:
    return XAUGoldStrategy(
        config=config,
        db_manager=db_manager,
        logger=logger,
        config_override=config_override,
    )
