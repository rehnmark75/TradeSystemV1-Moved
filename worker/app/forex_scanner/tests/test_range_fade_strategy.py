"""Unit tests for the EURUSD range-fade prototype strategy."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from forex_scanner.core.strategies.range_fade_strategy import (
    EURUSDRangeFadeStrategy,
    apply_config_overrides,
)
from forex_scanner.services.range_fade_config_service import (
    EURUSD_EPIC,
    EURUSDRangeFadeConfig,
    EURUSDRangeFadeConfigService,
    build_range_fade_config,
)


def _make_5m_buy_setup(bars: int = 200) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp("2026-04-21 14:00", tz="UTC"), periods=bars, freq="5min")
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
        self.assertEqual(cfg.strategy_name, "RANGE_FADE")
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
        self.assertEqual(cfg.backtest_overrides["fixed_take_profit_pips"], 15)

    def test_backtest_overrides_take_precedence_over_pair_overrides(self):
        cfg = EURUSDRangeFadeConfig(
            pair_overrides={
                EURUSD_EPIC: {
                    "fixed_stop_loss_pips": 8,
                    "fixed_take_profit_pips": 12,
                }
            }
        )
        apply_config_overrides(cfg, {
            "fixed_stop_loss_pips": 5,
            "fixed_take_profit_pips": 9,
        })

        self.assertEqual(cfg.get_pair_fixed_stop_loss_pips(EURUSD_EPIC), 5)
        self.assertEqual(cfg.get_pair_fixed_take_profit_pips(EURUSD_EPIC), 9)

    def test_direction_allowlist_uses_overrides(self):
        cfg = EURUSDRangeFadeConfig()
        self.assertTrue(cfg.is_direction_allowed(EURUSD_EPIC, "BUY"))
        self.assertTrue(cfg.is_direction_allowed(EURUSD_EPIC, "SELL"))

        apply_config_overrides(cfg, {"allowed_directions": "SELL"})

        self.assertFalse(cfg.is_direction_allowed(EURUSD_EPIC, "BUY"))
        self.assertTrue(cfg.is_direction_allowed(EURUSD_EPIC, "SELL"))

    def test_dynamic_pair_parameter_overrides_control_hours_and_direction(self):
        cfg = EURUSDRangeFadeConfig(
            pair_overrides={
                EURUSD_EPIC: {
                    "parameter_overrides": {
                        "london_start_hour_utc": 8,
                        "new_york_end_hour_utc": 16,
                        "blocked_hours_utc": "12,15",
                        "buy_blocked_hours_utc": "10",
                        "sell_blocked_hours_utc": "6",
                        "buy_start_hour_utc": 6,
                        "buy_end_hour_utc": 13,
                        "buy_allowed_hours_utc": "9,10,11,12",
                        "sell_allowed_hours_utc": "8,9,11,13,14,15",
                        "buy_allowed_htf_biases": "bullish",
                        "sell_allowed_htf_biases": "bearish,neutral",
                        "allowed_directions": "SELL",
                    }
                }
            }
        )

        self.assertFalse(cfg.is_session_allowed(7, EURUSD_EPIC))
        self.assertTrue(cfg.is_session_allowed(8, EURUSD_EPIC))
        self.assertFalse(cfg.is_session_allowed(12, EURUSD_EPIC))
        self.assertFalse(cfg.is_session_allowed(15, EURUSD_EPIC))
        self.assertFalse(cfg.is_session_allowed(17, EURUSD_EPIC))
        self.assertFalse(cfg.is_direction_session_allowed(6, EURUSD_EPIC, "BUY"))
        self.assertFalse(cfg.is_direction_session_allowed(6, EURUSD_EPIC, "SELL"))
        self.assertTrue(cfg.is_direction_session_allowed(10, EURUSD_EPIC, "BUY"))
        self.assertFalse(cfg.is_direction_session_allowed(14, EURUSD_EPIC, "BUY"))
        self.assertTrue(cfg.is_direction_session_allowed(12, EURUSD_EPIC, "BUY"))
        self.assertTrue(cfg.is_direction_session_allowed(13, EURUSD_EPIC, "SELL"))
        self.assertFalse(cfg.is_direction_session_allowed(12, EURUSD_EPIC, "SELL"))
        self.assertTrue(cfg.is_htf_bias_allowed(EURUSD_EPIC, "BUY", "bullish"))
        self.assertFalse(cfg.is_htf_bias_allowed(EURUSD_EPIC, "BUY", "neutral"))
        self.assertTrue(cfg.is_htf_bias_allowed(EURUSD_EPIC, "SELL", "neutral"))
        self.assertFalse(cfg.is_direction_allowed(EURUSD_EPIC, "BUY"))
        self.assertTrue(cfg.is_direction_allowed(EURUSD_EPIC, "SELL"))

    def test_dynamic_pair_parameter_overrides_control_directional_adx(self):
        cfg = EURUSDRangeFadeConfig(
            adx_ceiling=25,
            pair_overrides={
                EURUSD_EPIC: {
                    "parameter_overrides": {
                        "buy_adx_ceiling": 50,
                        "sell_adx_ceiling": 24,
                    }
                }
            },
        )

        self.assertEqual(cfg.get_pair_adx_ceiling(EURUSD_EPIC, "BUY"), 50)
        self.assertEqual(cfg.get_pair_adx_ceiling(EURUSD_EPIC, "SELL"), 24)

    def test_build_5m_profile(self):
        cfg = build_range_fade_config("5m")
        self.assertEqual(cfg.strategy_name, "RANGE_FADE")
        self.assertEqual(cfg.profile_name, "5m")
        self.assertEqual(cfg.primary_timeframe, "5m")
        self.assertEqual(cfg.confirmation_timeframe, "1h")
        self.assertFalse(cfg.allow_neutral_htf)


class EURUSDRangeFadeStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        EURUSDRangeFadeConfigService._instance = None
        svc = EURUSDRangeFadeConfigService.get_instance()
        svc._cached = {}
        svc._cache_ts = {}
        svc._cached["5m"] = EURUSDRangeFadeConfig()
        svc._cache_ts["5m"] = datetime.now()
        self.strat = EURUSDRangeFadeStrategy()

    def test_strategy_name(self):
        self.assertEqual(self.strat.strategy_name, "RANGE_FADE")
        self.assertEqual(self.strat.get_required_timeframes(), ["1h", "5m"])

    def test_unsupported_pair_returns_none(self):
        # Range-fade now supports multiple pairs; ensure an epic outside
        # the configured enabled_pairs list is rejected.
        signal = self.strat.detect_signal(
            df_trigger=_make_5m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic="CS.D.NOTAPAIR.MINI.IP",
            pair="NOTAPAIR",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNone(signal)

    def test_buy_signal_generated_for_stretched_selloff_into_range_low(self):
        signal = self.strat.detect_signal(
            df_trigger=_make_5m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal["strategy"], "RANGE_FADE")
        self.assertEqual(signal["signal"], "BUY")
        # monitor_only is config/DB-driven; just confirm the field is present.
        self.assertIn("monitor_only", signal)
        self.assertGreaterEqual(signal["confidence"], self.strat.config.min_confidence)

    def test_session_filter_blocks_asian(self):
        signal = self.strat.detect_signal(
            df_trigger=_make_5m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 03:00", tz="UTC").to_pydatetime(),
        )
        self.assertIsNone(signal)

    def test_direction_filter_blocks_disallowed_buy(self):
        strat = EURUSDRangeFadeStrategy(config=EURUSDRangeFadeConfig(
            allowed_directions="SELL",
        ))

        signal = strat.detect_signal(
            df_trigger=_make_5m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )

        self.assertIsNone(signal)

    def test_directional_adx_ceiling_allows_high_adx_buy_override(self):
        strat = EURUSDRangeFadeStrategy(config=EURUSDRangeFadeConfig(
            buy_adx_ceiling=50,
            sell_adx_ceiling=25,
        ))
        df = _make_5m_buy_setup()
        df["adx"] = 40.0

        signal = strat.detect_signal(
            df_trigger=df,
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal["signal"], "BUY")
        self.assertEqual(signal["strategy_indicators"]["adx_ceiling"], 50)

    def test_macd_histogram_filter_blocks_low_momentum_setup(self):
        strat = EURUSDRangeFadeStrategy(config=EURUSDRangeFadeConfig(
            min_macd_histogram_pips=100.0,
        ))

        signal = strat.detect_signal(
            df_trigger=_make_5m_buy_setup(),
            df_4h=_make_1h_bullish_htf(),
            epic=EURUSD_EPIC,
            pair="EURUSD",
            current_timestamp=pd.Timestamp("2026-04-21 14:00", tz="UTC").to_pydatetime(),
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
        self.assertEqual(strat.strategy_name, "RANGE_FADE")
        self.assertEqual(strat.config.primary_timeframe, "5m")
        self.assertEqual(strat.get_required_timeframes(), ["1h", "5m"])

    def test_dynamic_sl_tp_uses_band_width_clamps(self):
        strat = EURUSDRangeFadeStrategy(config=EURUSDRangeFadeConfig(
            dynamic_sl_tp_enabled=True,
            dynamic_sl_band_width_sl_mult=0.55,
            dynamic_sl_band_width_tp_mult=0.85,
            dynamic_sl_min_pips=5.0,
            dynamic_sl_max_pips=9.0,
            dynamic_tp_min_pips=8.0,
            dynamic_tp_max_pips=15.0,
        ))

        self.assertEqual(strat._resolve_sl_tp_pips(strat.config, EURUSD_EPIC, 4.0), (5.0, 8.0))
        self.assertEqual(strat._resolve_sl_tp_pips(strat.config, EURUSD_EPIC, 12.0), (6.6, 10.2))
        self.assertEqual(strat._resolve_sl_tp_pips(strat.config, EURUSD_EPIC, 30.0), (9.0, 15.0))


if __name__ == "__main__":
    unittest.main()
