#!/usr/bin/env python3
"""
Fix imports after moving backtest files to backtests/ directory
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    if not file_path.endswith('.py'):
        return False

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        original_content = content

        # Fix various import patterns
        replacements = [
            # Main backtest modules that moved to backtests/
            (r'from core\.backtest\.performance_analyzer import', 'from performance_analyzer import'),
            (r'from core\.backtest\.signal_analyzer import', 'from signal_analyzer import'),
            (r'from core\.backtest\.unified_backtest_engine import', 'from unified_backtest_engine import'),
            (r'from core\.backtest\.backtest_config import', 'from backtest_config import'),
            (r'from core\.backtest\.parameter_manager import', 'from parameter_manager import'),
            (r'from core\.backtest\.strategy_registry import', 'from strategy_registry import'),
            (r'from core\.backtest\.report_generator import', 'from report_generator import'),

            # Fallback imports that need updating
            (r'from forex_scanner\.core\.backtest\.performance_analyzer import', 'from forex_scanner.backtests.performance_analyzer import'),
            (r'from forex_scanner\.core\.backtest\.signal_analyzer import', 'from forex_scanner.backtests.signal_analyzer import'),
            (r'from forex_scanner\.core\.backtest\.unified_backtest_engine import', 'from forex_scanner.backtests.unified_backtest_engine import'),
            (r'from forex_scanner\.core\.backtest\.backtest_config import', 'from forex_scanner.backtests.backtest_config import'),
            (r'from forex_scanner\.core\.backtest\.parameter_manager import', 'from forex_scanner.backtests.parameter_manager import'),
            (r'from forex_scanner\.core\.backtest\.strategy_registry import', 'from forex_scanner.backtests.strategy_registry import'),
            (r'from forex_scanner\.core\.backtest\.report_generator import', 'from forex_scanner.backtests.report_generator import'),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix imports in all Python files in backtests directory"""
    backtests_dir = "/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/backtests"

    updated_files = []

    for root, dirs, files in os.walk(backtests_dir):
        for file in files:
            if file.endswith('.py') and file != 'fix_imports.py':
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    updated_files.append(file_path)

    print(f"Updated imports in {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()