#!/usr/bin/env python3
"""
RANGE_STRUCTURE Config Service — DB-backed configuration with 5-minute TTL cache.

Source of truth: `strategy_config` database
    - range_structure_global_config    (single is_active=TRUE row)
    - range_structure_pair_overrides   (per-epic nullable overrides)

Pattern follows `services.smc_simple_config_service` (singleton with TTL cache)
and mirrors the dataclass-based shape used in `mean_reversion_strategy`.

Usage:
    from forex_scanner.services.range_structure_config_service import (
        get_range_structure_config,
    )

    cfg = get_range_structure_config()
    sl_min = cfg.get_pair_sl_pips_min('CS.D.USDJPY.MINI.IP')
    if not cfg.is_pair_enabled(epic):
        return None
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras


# ---------------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class RangeStructureConfig:
    """In-memory snapshot of range_structure_global_config plus pair overrides."""

    # Identity
    strategy_name: str = "RANGE_STRUCTURE"
    strategy_version: str = "1.0.0"

    # Range build / sweep detection
    range_lookback_bars: int = 40
    sweep_penetration_pips: float = 1.5
    rejection_wick_ratio: float = 0.60

    # Confluence / targets
    ob_fvg_confluence_required: bool = True
    equilibrium_target_enabled: bool = True

    # R:R floor
    min_rr_ratio: float = 1.33

    # Hard ADX gates
    adx_hard_ceiling_primary: float = 20.0
    adx_hard_ceiling_htf: float = 22.0
    adx_period: int = 14

    # SL/TP clamps (pips)
    sl_pips_min: float = 6.0
    sl_pips_max: float = 12.0
    tp_pips_min: float = 10.0
    tp_pips_max: float = 18.0
    sl_buffer_pips: float = 1.0

    # HTF bias band: |score - 0.5| <= htf_bias_neutral_band
    htf_bias_neutral_band: float = 0.40

    # Cooldown / confidence
    signal_cooldown_minutes: int = 60
    min_confidence: float = 0.55
    max_confidence: float = 0.80

    # Routing / timeframes
    trust_regime_routing: bool = True
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"

    # Pair overrides, keyed by epic.
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Per-pair accessors
    # ------------------------------------------------------------------

    def _get_for_pair(self, epic: str, param_name: str, default: Any) -> Any:
        """Pair direct-column → JSONB parameter_overrides → global default."""
        if epic in self._pair_overrides:
            row = self._pair_overrides[epic]
            if param_name in row and row[param_name] is not None:
                return row[param_name]
            jsonb = row.get("parameter_overrides") or {}
            if isinstance(jsonb, dict) and param_name in jsonb and jsonb[param_name] is not None:
                return jsonb[param_name]
        return default

    # --- gate parameters ---
    def get_pair_adx_hard_ceiling_primary(self, epic: str) -> float:
        return float(self._get_for_pair(
            epic, "adx_hard_ceiling_primary", self.adx_hard_ceiling_primary))

    def get_pair_adx_hard_ceiling_htf(self, epic: str) -> float:
        return float(self._get_for_pair(
            epic, "adx_hard_ceiling_htf", self.adx_hard_ceiling_htf))

    def get_pair_sweep_penetration_pips(self, epic: str) -> float:
        return float(self._get_for_pair(
            epic, "sweep_penetration_pips", self.sweep_penetration_pips))

    def get_pair_rejection_wick_ratio(self, epic: str) -> float:
        """Clamped at 0.55 from below — do NOT relax below this floor (see plan)."""
        raw = float(self._get_for_pair(
            epic, "rejection_wick_ratio", self.rejection_wick_ratio))
        return max(0.55, raw)

    def get_pair_range_lookback_bars(self, epic: str) -> int:
        return int(self._get_for_pair(
            epic, "range_lookback_bars", self.range_lookback_bars))

    def get_pair_ob_fvg_confluence_required(self, epic: str) -> bool:
        return bool(self._get_for_pair(
            epic, "ob_fvg_confluence_required", self.ob_fvg_confluence_required))

    # --- risk ---
    def get_pair_sl_pips_min(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "sl_pips_min", self.sl_pips_min))

    def get_pair_sl_pips_max(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "sl_pips_max", self.sl_pips_max))

    def get_pair_tp_pips_min(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "tp_pips_min", self.tp_pips_min))

    def get_pair_tp_pips_max(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "tp_pips_max", self.tp_pips_max))

    def get_pair_min_rr_ratio(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "min_rr_ratio", self.min_rr_ratio))

    # --- confidence / cooldown ---
    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "min_confidence", self.min_confidence))

    def get_pair_max_confidence(self, epic: str) -> float:
        return float(self._get_for_pair(epic, "max_confidence", self.max_confidence))

    def get_pair_signal_cooldown_minutes(self, epic: str) -> int:
        return int(self._get_for_pair(
            epic, "signal_cooldown_minutes", self.signal_cooldown_minutes))

    # --- flags ---
    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return bool(self._pair_overrides[epic].get("is_enabled", True))
        # Unknown pairs default to disabled — explicit opt-in model.
        return False

    def is_pair_monitor_only(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            v = self._pair_overrides[epic].get("monitor_only")
            if v is None:
                return True
            return bool(v)
        return True

    def is_pair_traded(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return bool(self._pair_overrides[epic].get("is_traded", False))
        return False


# ---------------------------------------------------------------------------
# SERVICE (singleton with 5-min TTL cache)
# ---------------------------------------------------------------------------

class RangeStructureConfigService:
    """Singleton config service. Reloads from DB on `refresh()` or after TTL."""

    _instance: Optional["RangeStructureConfigService"] = None
    _cache_ttl_seconds: int = 300

    def __init__(self) -> None:
        self._config: Optional[RangeStructureConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._db_manager = None
        self._logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> "RangeStructureConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_db_manager(self, db_manager) -> None:
        """Optional: inject a db_manager (unused today; reserved for future
        shared-connection pooling). We currently open our own psycopg2
        connection against STRATEGY_CONFIG_DATABASE_URL."""
        self._db_manager = db_manager

    def get_config(self) -> RangeStructureConfig:
        now = datetime.now()
        stale = (
            self._config is None
            or self._last_refresh is None
            or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds
        )
        if stale:
            self._config = self._load_from_database()
            self._last_refresh = now
        assert self._config is not None  # narrow for type checker
        return self._config

    def refresh(self) -> RangeStructureConfig:
        """Force an immediate reload from the database."""
        self._config = None
        self._last_refresh = None
        return self.get_config()

    # ------------------------------------------------------------------
    # INTERNAL LOADER
    # ------------------------------------------------------------------

    def _load_from_database(self) -> RangeStructureConfig:
        config = RangeStructureConfig()
        database_url = os.getenv(
            "STRATEGY_CONFIG_DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/strategy_config",
        )

        try:
            conn = psycopg2.connect(database_url)
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

                # Global row
                cur.execute(
                    """
                    SELECT * FROM range_structure_global_config
                     WHERE is_active = TRUE
                     ORDER BY id DESC LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row:
                    for key in row.keys():
                        if not hasattr(config, key):
                            continue
                        v = row[key]
                        if v is None:
                            continue
                        default_val = getattr(RangeStructureConfig, key, None)
                        if isinstance(default_val, bool):
                            v = bool(v)
                        elif isinstance(default_val, int) and not isinstance(default_val, bool):
                            v = int(v)
                        elif isinstance(default_val, float):
                            v = float(v)
                        setattr(config, key, v)

                # Per-pair overrides
                cur.execute("SELECT * FROM range_structure_pair_overrides")
                config._pair_overrides = {r["epic"]: dict(r) for r in cur.fetchall()}

                cur.close()

                self._logger.info(
                    "[RANGE_STRUCTURE] Config loaded v%s | ADX gates %.1f/%.1f | "
                    "SL [%.1f,%.1f]p TP [%.1f,%.1f]p | %d pair overrides",
                    config.strategy_version,
                    config.adx_hard_ceiling_primary, config.adx_hard_ceiling_htf,
                    config.sl_pips_min, config.sl_pips_max,
                    config.tp_pips_min, config.tp_pips_max,
                    len(config._pair_overrides),
                )
            finally:
                conn.close()
        except Exception as e:
            self._logger.warning(
                "[RANGE_STRUCTURE] DB load failed — using dataclass defaults: %s", e
            )

        return config


# ---------------------------------------------------------------------------
# MODULE-LEVEL ACCESSOR
# ---------------------------------------------------------------------------

def get_range_structure_config() -> RangeStructureConfig:
    """Convenience accessor — equivalent to
    `RangeStructureConfigService.get_instance().get_config()`."""
    return RangeStructureConfigService.get_instance().get_config()
