"""Unit tests for XAU_GOLD strategy."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forex_scanner.core.strategies.xau_gold_strategy import XAUGoldStrategy
from forex_scanner.services.xau_gold_config_service import (
    XAUGoldConfig,
    XAUGoldConfigService,
)


GOLD_EPIC = "CS.D.CFEGOLD.CEE.IP"


def _make_htf(trend: str = "up", bars: int = 260, start_price: float = 1900.0) -> pd.DataFrame:
    """Build a deterministic 4H DataFrame with a clean trend."""
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 0.5, size=bars)  # small noise
    drift = 0.5 if trend == "up" else -0.5
    prices = start_price + np.cumsum(steps + drift)
    idx = pd.date_range(end=pd.Timestamp("2026-04-01 00:00", tz="UTC"), periods=bars, freq="4H")
    highs = prices + 1.5
    lows = prices - 1.5
    opens = np.concatenate([[prices[0]], prices[:-1]])
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices}, index=idx
    )


def _make_trigger_with_bos(bars: int = 80, bias: str = "bullish") -> pd.DataFrame:
    """1H trigger with a clear recent breakout."""
    base = 1900.0
    # First 70 bars flat range 1895-1905
    idx = pd.date_range(end=pd.Timestamp("2026-04-01 12:00", tz="UTC"), periods=bars, freq="1H")
    rng = np.random.default_rng(7)
    body = base + rng.uniform(-5, 5, size=bars)
    if bias == "bullish":
        body[-1] = base + 15  # breakout close above recent high
    else:
        body[-1] = base - 15
    return pd.DataFrame(
        {
            "open": body,
            "high": body + 1.5,
            "low": body - 1.5,
            "close": body,
        },
        index=idx,
    )


def _make_entry_pullback(bos_from: float, bos_to: float, bias: str = "bullish", bars: int = 40) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp("2026-04-01 12:00", tz="UTC"), periods=bars, freq="15min")
    leg = bos_to - bos_from
    # Place last close at fib 0.5 pullback
    target = bos_to - leg * 0.5
    closes = np.linspace(bos_to, target, bars)
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes},
        index=idx,
    )


class XAUGoldConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use pure in-memory config to keep tests hermetic
        self.cfg = XAUGoldConfig()

    def test_defaults(self):
        self.assertEqual(self.cfg.htf_timeframe, "4h")
        self.assertEqual(self.cfg.trigger_timeframe, "1h")
        self.assertEqual(self.cfg.entry_timeframe, "15m")
        self.assertAlmostEqual(self.cfg.rr_ratio, 2.0)
        self.assertTrue(self.cfg.block_ranging)

    def test_session_filter(self):
        # London block: 07-10, NY: 13-20, rollover 21-22 blocked
        self.assertTrue(self.cfg.is_session_allowed(8))   # London
        self.assertTrue(self.cfg.is_session_allowed(14))  # NY
        self.assertFalse(self.cfg.is_session_allowed(21)) # rollover
        self.assertFalse(self.cfg.is_session_allowed(3))  # Asian blocked
        self.cfg.asian_allowed = True
        self.assertTrue(self.cfg.is_session_allowed(3))

    def test_pair_monitor_only_default(self):
        self.cfg.pair_overrides[GOLD_EPIC] = {"monitor_only": True, "is_enabled": True, "pip_size": 0.1}
        self.assertTrue(self.cfg.is_monitor_only(GOLD_EPIC))
        self.assertEqual(self.cfg.get_pip_size(GOLD_EPIC), 0.1)


class XAUGoldStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset singleton so each test gets a fresh default config
        XAUGoldConfigService._instance = None
        svc = XAUGoldConfigService.get_instance()
        svc._cached = XAUGoldConfig()
        svc._cache_ts = datetime.now()
        svc._cached.pair_overrides[GOLD_EPIC] = {
            "is_enabled": True,
            "monitor_only": True,
            "pip_size": 0.1,
        }
        self.strat = XAUGoldStrategy()

    def test_strategy_name(self):
        self.assertEqual(self.strat.strategy_name, "XAU_GOLD")
        self.assertEqual(self.strat.get_required_timeframes(), ["4h", "1h", "15m"])

    def test_insufficient_data_returns_none(self):
        result = self.strat.detect_signal(
            df_trigger=pd.DataFrame(), df_4h=pd.DataFrame(), df_entry=pd.DataFrame(),
            epic=GOLD_EPIC, pair="XAUUSD",
        )
        self.assertIsNone(result)

    def test_disabled_pair_returns_none(self):
        XAUGoldConfigService.get_instance()._cached.enabled_pairs = []
        result = self.strat.detect_signal(
            df_trigger=_make_trigger_with_bos(), df_4h=_make_htf(),
            df_entry=_make_entry_pullback(1895, 1915), epic=GOLD_EPIC, pair="XAUUSD",
        )
        self.assertIsNone(result)

    def test_sl_tp_respects_floor_and_ratio(self):
        df = _make_trigger_with_bos()
        df = self.strat._enrich_trigger(df)
        sl_pips, tp_pips = self.strat._sl_tp(df, GOLD_EPIC)
        self.assertGreaterEqual(sl_pips, 25.0)
        self.assertLessEqual(sl_pips, 80.0)
        self.assertGreaterEqual(tp_pips / sl_pips, 1.99)

    def test_confidence_monotonic_and_capped(self):
        c = self.strat.config
        lo = self.strat._confidence(True, False, False, False, False)
        hi = self.strat._confidence(True, True, True, True, True)
        self.assertGreater(hi, lo)
        self.assertLessEqual(hi, c.max_confidence + 1e-9)

    def test_htf_bias_bullish(self):
        df = self.strat._enrich_htf(_make_htf("up"))
        self.assertEqual(self.strat._htf_bias(df), "bullish")

    def test_htf_bias_bearish(self):
        df = self.strat._enrich_htf(_make_htf("down"))
        self.assertEqual(self.strat._htf_bias(df), "bearish")


if __name__ == "__main__":
    unittest.main()
