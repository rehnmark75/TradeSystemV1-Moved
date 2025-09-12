# validation/__init__.py
"""
Signal Validation and Replay Module

This module provides comprehensive signal validation capabilities,
allowing users to replay historical scanning conditions and validate
why specific alerts were generated.

Components:
- HistoricalDataManager: Historical data retrieval and management
- ScannerStateRecreator: Recreation of historical scanner states
- ReplayEngine: Core orchestration of the replay process
- ValidationReporter: Detailed validation reporting
- SignalReplayValidator: Main entry point for validation operations

Usage:
    python -m forex_scanner.validation.signal_replay_validator \
        --timestamp "2025-01-15 14:30:00" \
        --epic "CS.D.EURUSD.MINI.IP"
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .historical_data_manager import HistoricalDataManager
from .scanner_state_recreator import ScannerStateRecreator
from .replay_engine import ReplayEngine
from .validation_reporter import ValidationReporter
from .signal_replay_validator import SignalReplayValidator

__all__ = [
    'HistoricalDataManager',
    'ScannerStateRecreator', 
    'ReplayEngine',
    'ValidationReporter',
    'SignalReplayValidator'
]