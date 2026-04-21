"""Unit tests for the EURUSD range-fade prototype strategy."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from forex_scanner.core.strategies.eurusd_range_fade_strategy import (
    EURUSD_EPIC,
    EURUSDRangeFadeConfig,
    EURUSDRangeFadeConfigService,
    EURUSDRangeFadeStrategy,
    apply_config_overrides,
    build_eurusd_range_fade_config,
)


def _make_15m_buy_setup(bars: int = 80) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp("2026-04-21 14:00", tz="UTC"), periods=bars, freq="15min")
    base = np.full(bars, 1.1000)
    wave = 0.0006 * np.sin(np.linspace(0, 10, bars))
    close = base + wave
    close[-6:] = [1.1001, 1.0999, 1.0996, 1.0993, 1.0990, 1.0987]
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.00025
    low = np.minimum(open_, close) - 0.00025
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df["adx"] = 18.0
    return df


def _make_1h_bullish_htf(bars: int = 120) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp("2026-04-21 14:00", tz="UTC"), periods=bars, freq="1h")
    close = np.linspace(1.0850, 1.1050, bars)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.0004
    low = np.minimum(open_, close) - 0.0004
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df["adx"] = 20.0
    return df


class EURUSDRangeFadeConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = EURUSDRangeFadeConfig()
        self.assertEqual(cfg.strategy_name, "EURUSD_RANGE_FADE")
        self.assertTrue(cfg.monitor_only)
        self.assertIn(EURUSD_EPIC, cfg.enabled_pairs)
        self.assertTrue(cfg.is_session_allowed(14))
        self.assertFalse(cfg.is_session_allowed(3))

    def test_apply_overrides(self):
        cfg = EURUSDRangeFadeConfig()
        apply_config_overrides(cfg, {
            "monitor_only": False,
            "rsi_oversold": 28,
            "fixed_take_profit_pips": 15,
        })
        self.assertFalse(cfg.monitor_only)
        self.assertEqual(cfg.rsi_oversold, 28)
        self.assertEqual(cfg.fixed_take_profit_pips, 15)

    def test_build_5m_profile(self):
        cfg = build_eurusd_range_fade_config("5m")
        self.assertEqual(cfg.strategy_name, "EURUSD_RANGE_FADE_5M")
        self.assertEqual(cfg.profile_name, "5m")
        self.assertEqual(cfg.primary_timeframe, "5m")
        self.assertEqual(cfg.confirmation_timeframe, "1h")
        self.assertFalse(cfg.allow_neutral_htf)


class EURUSDRangeFadeStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        EURUSDRangeFadeConfigService._instance = None
        svc = EURUSDRangeFadeConfigService.get_instance()
        svc._cached = EURUSDRangeFadeConfig()
        svc._cache_ts = datetime.now()
        self.strat = EURUSDRangeFadeStrategy()

    def test_strategy_name(self):
        self.assertEqual(self.strat.strategy_name, "EURUSD_RANGE_FADE")
        self.assertEqual(self.strat.get_required_timeframes(), ["1h", "15m"])

    def test_non_eurusd_pair_returns_none(self):
        signal = self.strat.detect_signal(
            df_trigger=_make_15m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic="CS.D.GBPUSD.MINI.IP",
            pair="GBPUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNone(signal)

    def test_buy_signal_generated_for_stretched_selloff_into_range_low(self):
        signal = self.strat.detect_signal(
            df_trigger=_make_15m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["strategy"], "EURUSD_RANGE_FADE")
        self.assertEqual(signal["signal"], "BUY")
        self.assertTrue(signal["monitor_only"])
        self.assertGreaterEqual(signal["confidence"], self.strat.config.min_confidence)

    def test_session_filter_blocks_asian(self):
        signal = self.strat.detect_signal(
            df_trigger=_make_15m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 03:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNone(signal)

    def test_constructor_honors_config_overrides(self):
        strat = EURUSDRangeFadeStrategy(config_override={
            "monitor_only": False,
            "fixed_stop_loss_pips": 11,
            "fixed_take_profit_pips": 17,
        })
        self.assertFalse(strat.config.monitor_only)
        self.assertEqual(strat.config.fixed_stop_loss_pips, 11)
        self.assertEqual(strat.config.fixed_take_profit_pips, 17)

    def test_constructor_can_select_5m_profile(self):
        strat = EURUSDRangeFadeStrategy(config_override={"erf_profile": "5m"})
        self.assertEqual(strat.strategy_name, "EURUSD_RANGE_FADE_5M")
        self.assertEqual(strat.config.primary_timeframe, "5m")
        self.assertEqual(strat.get_required_timeframes(), ["1h", "5m"])


if __name__ == "__main__":
    unittest.main()
