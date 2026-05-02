# core/scanning/signal_detection_engine.py
"""
SignalDetectionEngine - Shared signal detection logic for live and backtest scanners.

Provides the minimal, stateless detection surface:
  - Iterate enabled epics
  - Delegate to signal_detector.detect_signals_all_strategies / _detect_single_strategy
  - Force-initialize strategies for backtest

Intentionally excludes: alert dedup, order management, Claude integration,
AlertHistoryManager, market intelligence capture, scan performance capture.
Those live exclusively in IntelligentForexScanner (live path).
"""

import logging
from typing import List, Optional, Dict


class SignalDetectionEngine:
    """
    Thin shared engine that both IntelligentForexScanner (via composition) and
    BacktestScanner use for pure signal detection.

    Parameters
    ----------
    db_manager : DatabaseManager
        Shared database connection.
    signal_detector : SignalDetector
        Fully initialised SignalDetector instance (may have its data_fetcher
        replaced by a BacktestDataFetcher before use).
    epic_list : list[str]
        Epics to iterate when detect_for_all_epics is called.
    spread_pips : float
        Spread in pips forwarded to detection calls.
    logger : logging.Logger | None
        If None a module-level logger is created.
    """

    def __init__(
        self,
        db_manager,
        signal_detector,
        epic_list: List[str],
        spread_pips: float = 1.5,
        logger: Optional[logging.Logger] = None,
    ):
        self.db_manager = db_manager
        self.signal_detector = signal_detector
        self.epic_list = epic_list
        self.spread_pips = spread_pips
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Strategy initialisation helpers
    # ------------------------------------------------------------------

    def force_initialize_strategy(self, strategy_name: str):
        """
        Delegate to signal_detector.force_initialize_strategy.
        Returns (success: bool, message: str).
        """
        if not hasattr(self.signal_detector, 'force_initialize_strategy'):
            return False, "signal_detector does not support force_initialize_strategy"
        return self.signal_detector.force_initialize_strategy(strategy_name)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def detect_all_strategies(
        self,
        epic: str,
        timeframe: str = '15m',
    ) -> Optional[List[Dict]]:
        """
        Run all active strategies for a single epic.
        Returns a list of signal dicts (may be empty) or None.
        """
        pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '').replace('.IP', '')
        return self.signal_detector.detect_signals_all_strategies(
            epic, pair_name, self.spread_pips, timeframe
        )

    def detect_single_strategy(
        self,
        strategy_name: str,
        epic: str,
        timeframe: str = '15m',
        current_timestamp=None,
    ) -> Optional[Dict]:
        """
        Run a single named strategy for one epic.
        Returns a signal dict or None.
        """
        pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '').replace('.IP', '')
        kwargs = {}
        if current_timestamp is not None:
            kwargs['current_timestamp'] = current_timestamp
        return self.signal_detector._detect_single_strategy(
            strategy_name, epic, pair_name, self.spread_pips, timeframe, **kwargs
        )

    def detect_for_all_epics(
        self,
        strategy_name: str,
        timeframe: str = '15m',
        current_timestamp=None,
    ) -> List[Dict]:
        """
        Iterate self.epic_list and collect signals.
        strategy_name == 'ALL' / '' triggers all-strategies path.
        Returns flat list of signal dicts.
        """
        signals: List[Dict] = []
        run_all = strategy_name.upper() in ('ALL', 'ALL_STRATEGIES', '')

        for epic in self.epic_list:
            try:
                if run_all:
                    result = self.detect_all_strategies(epic, timeframe)
                else:
                    result = self.detect_single_strategy(
                        strategy_name, epic, timeframe, current_timestamp
                    )
                if result:
                    if isinstance(result, list):
                        signals.extend(result)
                    else:
                        signals.append(result)
            except Exception as exc:
                self.logger.error(f"SignalDetectionEngine: error for {epic}: {exc}")

        return signals
