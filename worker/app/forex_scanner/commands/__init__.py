# commands/__init__.py
"""
Command modules for Forex Scanner CLI
Modular command organization for better maintainability
"""

from .scanner_commands import ScannerCommands
from .backtest_commands import BacktestCommands
from .debug_commands import DebugCommands
from .claude_commands import ClaudeCommands
from .analysis_commands import AnalysisCommands

__all__ = [
    'ScannerCommands',
    'BacktestCommands',
    'DebugCommands',
    'ClaudeCommands',
    'AnalysisCommands'
]

__version__ = "1.0.0"
__author__ = "Forex Scanner Team"
