"""
TradingView Exporter

Exports signals to various formats compatible with TradingView:
1. CSV Watchlist - Import directly to TradingView watchlist
2. Pine Script Alerts - Generate alert conditions
3. Signal Table - Human-readable format

CSV Format:
Symbol,Entry,Stop,TP1,TP2,Score,Tier,Setup

Can be imported via TradingView's "Import List" feature.
"""

import csv
import io
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..base_scanner import SignalSetup, SignalType

logger = logging.getLogger(__name__)


class TradingViewExporter:
    """
    Exports scanner signals to TradingView-compatible formats.

    Usage:
        exporter = TradingViewExporter()

        # Export to CSV string
        csv_content = exporter.to_csv(signals)

        # Export to file
        exporter.to_csv_file(signals, "/path/to/watchlist.csv")

        # Generate Pine Script
        pine_script = exporter.to_pine_script(signals)

        # Create human-readable table
        table = exporter.to_markdown_table(signals)
    """

    # TradingView exchange prefixes
    EXCHANGE_MAP = {
        'default': 'NASDAQ',
        'NYSE': 'NYSE',
        'NASDAQ': 'NASDAQ',
        'AMEX': 'AMEX',
    }

    def __init__(self, default_exchange: str = 'NASDAQ'):
        """
        Initialize exporter.

        Args:
            default_exchange: Default exchange for symbols without exchange
        """
        self.default_exchange = default_exchange

    def to_csv(
        self,
        signals: List[SignalSetup],
        include_levels: bool = True,
        exchange: str = None
    ) -> str:
        """
        Export signals to CSV format for TradingView import.

        Args:
            signals: List of SignalSetup objects
            include_levels: Include entry/stop/target columns
            exchange: Exchange prefix (e.g., 'NASDAQ')

        Returns:
            CSV string content
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        if include_levels:
            writer.writerow([
                'Symbol', 'Exchange', 'Side', 'Entry', 'Stop', 'TP1', 'TP2',
                'Risk%', 'R:R', 'Score', 'Tier', 'Scanner', 'Setup'
            ])
        else:
            writer.writerow(['Symbol', 'Exchange', 'Score', 'Tier', 'Scanner'])

        # Write signals
        for signal in signals:
            symbol = signal.ticker
            exch = exchange or self.default_exchange

            if include_levels:
                writer.writerow([
                    symbol,
                    exch,
                    'LONG' if signal.signal_type == SignalType.BUY else 'SHORT',
                    f'{float(signal.entry_price):.2f}',
                    f'{float(signal.stop_loss):.2f}',
                    f'{float(signal.take_profit_1):.2f}',
                    f'{float(signal.take_profit_2):.2f}' if signal.take_profit_2 else '',
                    f'{float(signal.risk_percent):.1f}%',
                    f'{float(signal.risk_reward_ratio):.1f}',
                    signal.composite_score,
                    signal.quality_tier.value,
                    signal.scanner_name,
                    signal.setup_description[:50] if signal.setup_description else ''
                ])
            else:
                writer.writerow([
                    symbol,
                    exch,
                    signal.composite_score,
                    signal.quality_tier.value,
                    signal.scanner_name
                ])

        return output.getvalue()

    def to_csv_file(
        self,
        signals: List[SignalSetup],
        file_path: str,
        include_levels: bool = True
    ) -> bool:
        """
        Export signals to CSV file.

        Args:
            signals: List of SignalSetup objects
            file_path: Output file path
            include_levels: Include entry/stop/target columns

        Returns:
            True if successful
        """
        try:
            content = self.to_csv(signals, include_levels)
            Path(file_path).write_text(content)
            logger.info(f"Exported {len(signals)} signals to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False

    def to_watchlist_txt(self, signals: List[SignalSetup]) -> str:
        """
        Export as simple symbol list (TradingView watchlist format).

        This is the simplest format - just symbols, one per line.

        Returns:
            Text with one symbol per line
        """
        lines = []
        for signal in signals:
            # Format: EXCHANGE:SYMBOL
            lines.append(f"{self.default_exchange}:{signal.ticker}")
        return '\n'.join(lines)

    def to_pine_script(
        self,
        signals: List[SignalSetup],
        include_alerts: bool = True
    ) -> str:
        """
        Generate Pine Script for alerts and levels.

        Creates a Pine Script that:
        - Draws entry, stop, and target levels
        - Sets up alert conditions for each signal
        - Shows signal information in labels

        Args:
            signals: List of SignalSetup objects
            include_alerts: Include alert conditions

        Returns:
            Pine Script code
        """
        script_lines = [
            '//@version=5',
            'indicator("Signal Scanner Levels", overlay=true)',
            '',
            '// Signal data',
        ]

        # Create arrays for signals
        symbols = []
        entries = []
        stops = []
        tp1s = []
        scores = []
        tiers = []

        for signal in signals[:50]:  # Limit to 50 for Pine Script
            symbols.append(f'"{signal.ticker}"')
            entries.append(str(float(signal.entry_price)))
            stops.append(str(float(signal.stop_loss)))
            tp1s.append(str(float(signal.take_profit_1)))
            scores.append(str(signal.composite_score))
            tiers.append(f'"{signal.quality_tier.value}"')

        script_lines.extend([
            f'var string[] SYMBOLS = array.from({", ".join(symbols)})',
            f'var float[] ENTRIES = array.from({", ".join(entries)})',
            f'var float[] STOPS = array.from({", ".join(stops)})',
            f'var float[] TP1S = array.from({", ".join(tp1s)})',
            f'var int[] SCORES = array.from({", ".join(scores)})',
            f'var string[] TIERS = array.from({", ".join(tiers)})',
            '',
            '// Find current symbol in list',
            'currentSymbol = syminfo.ticker',
            'signalIndex = array.indexof(SYMBOLS, currentSymbol)',
            '',
            '// Draw levels if symbol found',
            'if signalIndex >= 0',
            '    entryPrice = array.get(ENTRIES, signalIndex)',
            '    stopPrice = array.get(STOPS, signalIndex)',
            '    tp1Price = array.get(TP1S, signalIndex)',
            '    score = array.get(SCORES, signalIndex)',
            '    tier = array.get(TIERS, signalIndex)',
            '',
            '    // Entry line (green)',
            '    line.new(bar_index - 50, entryPrice, bar_index, entryPrice,',
            '             color=color.green, width=2, extend=extend.right)',
            '',
            '    // Stop line (red)',
            '    line.new(bar_index - 50, stopPrice, bar_index, stopPrice,',
            '             color=color.red, width=2, style=line.style_dashed, extend=extend.right)',
            '',
            '    // TP1 line (blue)',
            '    line.new(bar_index - 50, tp1Price, bar_index, tp1Price,',
            '             color=color.blue, width=2, style=line.style_dotted, extend=extend.right)',
            '',
            '    // Info label',
            '    label.new(bar_index, entryPrice,',
            '              text="Score: " + str.tostring(score) + " | Tier: " + tier,',
            '              style=label.style_label_left, color=color.new(color.green, 80))',
            '',
        ])

        if include_alerts:
            script_lines.extend([
                '// Alert conditions',
                'alertcondition(signalIndex >= 0 and close >= array.get(ENTRIES, signalIndex),',
                '               title="Entry Hit", message="Signal entry level reached")',
                '',
                'alertcondition(signalIndex >= 0 and close <= array.get(STOPS, signalIndex),',
                '               title="Stop Hit", message="Signal stop loss reached")',
                '',
                'alertcondition(signalIndex >= 0 and close >= array.get(TP1S, signalIndex),',
                '               title="TP1 Hit", message="Signal TP1 reached")',
            ])

        return '\n'.join(script_lines)

    def to_markdown_table(
        self,
        signals: List[SignalSetup],
        include_factors: bool = False
    ) -> str:
        """
        Generate markdown table of signals.

        Args:
            signals: List of SignalSetup objects
            include_factors: Include confluence factors column

        Returns:
            Markdown table string
        """
        lines = []

        # Header
        if include_factors:
            lines.append(
                '| # | Symbol | Score | Tier | Entry | Stop | TP1 | R:R | Scanner | Factors |'
            )
            lines.append(
                '|---|--------|-------|------|-------|------|-----|-----|---------|---------|'
            )
        else:
            lines.append(
                '| # | Symbol | Score | Tier | Entry | Stop | TP1 | Risk% | R:R | Scanner |'
            )
            lines.append(
                '|---|--------|-------|------|-------|------|-----|-------|-----|---------|'
            )

        # Rows
        for i, signal in enumerate(signals, 1):
            if include_factors:
                factors = ', '.join(signal.confluence_factors[:3])
                lines.append(
                    f'| {i} | **{signal.ticker}** | {signal.composite_score} | '
                    f'{signal.quality_tier.value} | ${float(signal.entry_price):.2f} | '
                    f'${float(signal.stop_loss):.2f} | ${float(signal.take_profit_1):.2f} | '
                    f'{float(signal.risk_reward_ratio):.1f} | {signal.scanner_name} | {factors} |'
                )
            else:
                lines.append(
                    f'| {i} | **{signal.ticker}** | {signal.composite_score} | '
                    f'{signal.quality_tier.value} | ${float(signal.entry_price):.2f} | '
                    f'${float(signal.stop_loss):.2f} | ${float(signal.take_profit_1):.2f} | '
                    f'{float(signal.risk_percent):.1f}% | {float(signal.risk_reward_ratio):.1f} | '
                    f'{signal.scanner_name} |'
                )

        return '\n'.join(lines)

    def to_json(
        self,
        signals: List[SignalSetup],
        indent: int = 2
    ) -> str:
        """
        Export signals to JSON format.

        Args:
            signals: List of SignalSetup objects
            indent: JSON indentation

        Returns:
            JSON string
        """
        data = {
            'export_time': datetime.now().isoformat(),
            'signal_count': len(signals),
            'signals': [signal.to_tradingview_dict() for signal in signals]
        }
        return json.dumps(data, indent=indent, default=str)

    def to_telegram_message(
        self,
        signals: List[SignalSetup],
        max_signals: int = 10
    ) -> str:
        """
        Generate Telegram-formatted message.

        Args:
            signals: List of SignalSetup objects
            max_signals: Maximum signals to include

        Returns:
            Telegram markdown message
        """
        lines = [
            f"*Signal Scanner Report*",
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "",
            f"Total Signals: {len(signals)}",
            f"High Quality (A/A+): {sum(1 for s in signals if s.is_high_quality)}",
            "",
            "*Top Signals:*",
        ]

        for i, signal in enumerate(signals[:max_signals], 1):
            emoji = self._get_tier_emoji(signal.quality_tier.value)
            lines.append(
                f"{i}. {emoji} *{signal.ticker}* ({signal.quality_tier.value})"
            )
            lines.append(
                f"   Entry: ${float(signal.entry_price):.2f} | "
                f"Stop: ${float(signal.stop_loss):.2f} | "
                f"TP: ${float(signal.take_profit_1):.2f}"
            )
            lines.append(f"   Score: {signal.composite_score} | {signal.scanner_name}")
            lines.append("")

        return '\n'.join(lines)

    def _get_tier_emoji(self, tier: str) -> str:
        """Get emoji for quality tier"""
        tier_emojis = {
            'A+': '🔥',
            'A': '⭐',
            'B': '✓',
            'C': '•',
            'D': '◦'
        }
        return tier_emojis.get(tier, '•')

    def generate_daily_report(
        self,
        signals: List[SignalSetup],
        scan_stats: Dict[str, Any] = None
    ) -> str:
        """
        Generate comprehensive daily report.

        Args:
            signals: List of SignalSetup objects
            scan_stats: Statistics from scan run

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"  DAILY SIGNAL SCANNER REPORT",
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]

        # Summary stats
        if scan_stats:
            lines.extend([
                "SCAN SUMMARY",
                "-" * 40,
                f"  Total Signals: {scan_stats.get('total_signals', len(signals))}",
                f"  High Quality (A/A+): {scan_stats.get('high_quality_count', 0)}",
                f"  Scan Duration: {scan_stats.get('scan_duration_seconds', 0)}s",
                "",
            ])

        # Signals by scanner
        by_scanner = {}
        for signal in signals:
            scanner = signal.scanner_name
            if scanner not in by_scanner:
                by_scanner[scanner] = []
            by_scanner[scanner].append(signal)

        for scanner_name, scanner_signals in by_scanner.items():
            lines.extend([
                f"{scanner_name.upper().replace('_', ' ')}",
                "-" * 40,
            ])

            for signal in scanner_signals[:5]:  # Top 5 per scanner
                lines.append(
                    f"  {signal.quality_tier.value} | {signal.ticker:6} | "
                    f"Score: {signal.composite_score:3} | "
                    f"Entry: ${float(signal.entry_price):7.2f} | "
                    f"Stop: ${float(signal.stop_loss):7.2f} | "
                    f"R:R {float(signal.risk_reward_ratio):.1f}"
                )

            if len(scanner_signals) > 5:
                lines.append(f"  ... and {len(scanner_signals) - 5} more")

            lines.append("")

        # High quality summary
        high_quality = [s for s in signals if s.is_high_quality]
        if high_quality:
            lines.extend([
                "TOP A/A+ SIGNALS",
                "-" * 40,
            ])
            for signal in high_quality[:10]:
                factors = ', '.join(signal.confluence_factors[:3])
                lines.append(
                    f"  {signal.quality_tier.value} | {signal.ticker:6} | "
                    f"Score: {signal.composite_score} | {factors}"
                )
            lines.append("")

        lines.append("=" * 70)

        return '\n'.join(lines)
