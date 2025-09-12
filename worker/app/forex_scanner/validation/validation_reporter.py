# validation/validation_reporter.py
"""
Validation Reporter for Signal Validation

This module handles the generation of detailed validation reports,
providing comprehensive information about signal replay results,
market conditions, and decision processes.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import json
from dataclasses import dataclass

from .replay_config import OUTPUT_CONFIG


@dataclass
class ValidationResult:
    """Container for validation results"""
    success: bool
    epic: str
    timestamp: datetime
    signal_detected: bool
    signal_data: Optional[Dict[str, Any]] = None
    market_state: Optional[Dict[str, Any]] = None
    decision_path: Optional[List[Dict[str, str]]] = None
    comparison_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


class ValidationReporter:
    """
    Generates detailed validation reports for signal replay results
    
    This class provides methods to:
    - Format validation results into human-readable reports
    - Compare replayed signals with stored historical alerts
    - Generate summary statistics for batch validations
    - Export results to various formats (text, JSON, HTML)
    """
    
    def __init__(self, use_colors: bool = True, max_line_length: int = 80):
        """
        Initialize the validation reporter
        
        Args:
            use_colors: Whether to use ANSI colors in output
            max_line_length: Maximum line length for text formatting
        """
        self.logger = logging.getLogger(__name__)
        self.use_colors = use_colors
        self.max_line_length = max_line_length
        self.decimal_places = OUTPUT_CONFIG['decimal_places']
        self.symbols = OUTPUT_CONFIG['symbols']
        
        if use_colors:
            self.colors = OUTPUT_CONFIG['colors']
        else:
            # Disable colors
            self.colors = {k: '' for k in OUTPUT_CONFIG['colors'].keys()}
    
    def generate_validation_report(
        self,
        result: ValidationResult,
        show_calculations: bool = True,
        show_raw_data: bool = False,
        show_intermediate_steps: bool = False
    ) -> str:
        """
        Generate a comprehensive validation report for a single result
        
        Args:
            result: Validation result to report on
            show_calculations: Whether to show detailed calculations
            show_raw_data: Whether to include raw market data
            show_intermediate_steps: Whether to show intermediate decision steps
            
        Returns:
            Formatted validation report as string
        """
        try:
            lines = []
            
            # Header
            lines.extend(self._generate_header(result))
            lines.append("")
            
            # Market conditions
            if result.market_state:
                lines.extend(self._generate_market_conditions_section(result.market_state, show_raw_data))
                lines.append("")
            
            # Signal analysis
            if result.signal_detected:
                lines.extend(self._generate_signal_analysis_section(result, show_calculations, show_intermediate_steps))
                lines.append("")
            else:
                lines.extend(self._generate_no_signal_section(result))
                lines.append("")
            
            # Decision path
            if result.decision_path and show_intermediate_steps:
                lines.extend(self._generate_decision_path_section(result.decision_path))
                lines.append("")
            
            # Removed misleading comparison and final result sections
            
            # Footer
            lines.extend(self._generate_footer(result))
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating validation report: {e}")
            return f"❌ Error generating report: {str(e)}"
    
    def generate_batch_summary(
        self,
        results: List[ValidationResult],
        show_statistics: bool = True,
        show_failures: bool = True
    ) -> str:
        """
        Generate a summary report for batch validation results
        
        Args:
            results: List of validation results
            show_statistics: Whether to show detailed statistics
            show_failures: Whether to show failed validations
            
        Returns:
            Formatted batch summary report
        """
        try:
            lines = []
            
            # Header
            lines.append(self._colorize("🔍 Batch Validation Summary", 'info'))
            lines.append("=" * self.max_line_length)
            lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"📊 Total Validations: {len(results)}")
            lines.append("")
            
            # Overall statistics
            if show_statistics:
                lines.extend(self._generate_batch_statistics(results))
                lines.append("")
            
            # Results by epic
            lines.extend(self._generate_epic_summary(results))
            lines.append("")
            
            # Failed validations
            if show_failures:
                failed_results = [r for r in results if not r.success]
                if failed_results:
                    lines.extend(self._generate_failure_summary(failed_results))
                    lines.append("")
            
            # Performance summary
            lines.extend(self._generate_performance_summary(results))
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating batch summary: {e}")
            return f"❌ Error generating batch summary: {str(e)}"
    
    def export_to_json(self, results: Union[ValidationResult, List[ValidationResult]]) -> str:
        """
        Export validation results to JSON format
        
        Args:
            results: Single result or list of results to export
            
        Returns:
            JSON string representation of results
        """
        try:
            if isinstance(results, ValidationResult):
                results = [results]
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': []
            }
            
            for result in results:
                result_data = {
                    'success': result.success,
                    'epic': result.epic,
                    'timestamp': result.timestamp.isoformat(),
                    'signal_detected': result.signal_detected,
                    'processing_time_ms': result.processing_time_ms,
                    'signal_data': result.signal_data,
                    'market_state': result.market_state,
                    'decision_path': result.decision_path,
                    'comparison_result': result.comparison_result,
                    'error_message': result.error_message
                }
                export_data['results'].append(result_data)
            
            return json.dumps(export_data, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting to JSON: {e}")
            return f'{{"error": "Export failed: {str(e)}"}}'
    
    def _generate_header(self, result: ValidationResult) -> List[str]:
        """Generate report header"""
        lines = []
        
        lines.append(self._colorize("🔍 Signal Validation Report", 'info'))
        lines.append("=" * self.max_line_length)
        lines.append(f"📅 Target Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        lines.append(f"🎯 Epic: {result.epic}")
        
        if result.signal_data and 'strategy' in result.signal_data:
            lines.append(f"⚙️  Strategy: {result.signal_data['strategy']}")
        
        return lines
    
    def _generate_market_conditions_section(self, market_state: Dict[str, Any], show_raw_data: bool) -> List[str]:
        """Generate market conditions section"""
        lines = []
        
        lines.append(self._colorize("📊 Market Conditions at Timestamp:", 'info'))
        lines.append("-" * 40)
        
        # Price information
        if 'price' in market_state:
            price_data = market_state['price']
            lines.append(f"   Price: {self._format_price(price_data['close'])}")
            
            if show_raw_data:
                lines.append(f"   OHLC: {self._format_price(price_data['open'])}/{self._format_price(price_data['high'])}/{self._format_price(price_data['low'])}/{self._format_price(price_data['close'])}")
                if price_data.get('volume'):
                    lines.append(f"   Volume: {price_data['volume']:,}")
        
        # Technical indicators
        if 'indicators' in market_state:
            indicators = market_state['indicators']
            
            # EMAs
            ema_indicators = {k: v for k, v in indicators.items() if 'ema' in k.lower()}
            if ema_indicators:
                ema_line = "   EMAs: "
                ema_parts = []
                for name, value in sorted(ema_indicators.items()):
                    if isinstance(value, (int, float)):
                        ema_parts.append(f"{name.upper()}: {self._format_price(value)}")
                lines.append(ema_line + " | ".join(ema_parts))
            
            # Other key indicators
            key_indicators = ['rsi', 'atr', 'macd', 'kama']
            for indicator in key_indicators:
                if indicator in indicators:
                    value = indicators[indicator]
                    lines.append(f"   {indicator.upper()}: {self._format_indicator(value)}")
        
        # Trend analysis
        if 'trend' in market_state:
            trend = market_state['trend']
            direction = trend.get('direction', 'UNKNOWN')
            color = 'success' if direction == 'BULLISH' else 'error' if direction == 'BEARISH' else 'warning'
            lines.append(f"   Trend: {self._colorize(direction, color)}")
            
            if 'ema_alignment' in trend:
                alignment_status = self._colorize("✓", 'success') if trend['ema_alignment'] else self._colorize("✗", 'error')
                lines.append(f"   EMA Alignment: {alignment_status}")
        
        # Volatility
        if 'conditions' in market_state and 'volatility' in market_state['conditions']:
            vol_data = market_state['conditions']['volatility']
            regime = vol_data.get('regime', 'UNKNOWN')
            lines.append(f"   Volatility: {regime} (ATR: {self._format_indicator(vol_data.get('atr', 0))})")
        
        return lines
    
    def _generate_signal_analysis_section(self, result: ValidationResult, show_calculations: bool, show_intermediate_steps: bool) -> List[str]:
        """Generate signal analysis section"""
        lines = []
        
        if not result.signal_data:
            return lines
        
        signal = result.signal_data
        
        lines.append(self._colorize("🔬 Signal Detection Analysis:", 'info'))
        lines.append("-" * 40)
        
        # Basic signal information
        signal_type = signal.get('signal_type', 'UNKNOWN')
        confidence = signal.get('confidence_score', 0)
        
        lines.append(f"   Signal Type: {self._colorize(signal_type, 'success' if signal_type in ['BULL', 'BUY'] else 'error')}")
        lines.append(f"   Confidence: {self._format_percentage(confidence)}")
        
        if 'entry_price' in signal:
            lines.append(f"   Entry Price: {self._format_price(signal['entry_price'])}")
        
        # Strategy-specific analysis
        strategy = signal.get('strategy', '').lower()
        
        if 'ema' in strategy:
            lines.extend(self._generate_ema_analysis(signal, show_calculations))
        elif 'macd' in strategy:
            lines.extend(self._generate_macd_analysis(signal, show_calculations))
        elif 'kama' in strategy:
            lines.extend(self._generate_kama_analysis(signal, show_calculations))
        
        # Show additional details if intermediate steps requested
        if show_intermediate_steps and signal.get('processing_metadata'):
            lines.append(f"   🔍 Processing Details: {len(signal['processing_metadata'])} metadata items")
        
        # Filters and validations
        lines.extend(self._generate_filter_analysis(signal))
        
        # Smart Money analysis
        if signal.get('smart_money_validated'):
            lines.extend(self._generate_smart_money_analysis(signal))
        
        return lines
    
    def _generate_no_signal_section(self, result: ValidationResult) -> List[str]:
        """Generate section for when no signal was detected"""
        lines = []
        
        lines.append(self._colorize("ℹ️  No Signal Detected", 'warning'))
        lines.append("-" * 40)
        
        if result.market_state:
            # Analyze why no signal was generated
            market_state = result.market_state
            
            # Check common signal requirements
            if 'trend' in market_state:
                trend = market_state['trend']
                if trend.get('direction') == 'SIDEWAYS':
                    lines.append(f"   {self._colorize('⚠️', 'warning')} Trend Direction: SIDEWAYS (signals require clear trend)")
                if not trend.get('ema_alignment', False):
                    lines.append(f"   {self._colorize('⚠️', 'warning')} EMA Alignment: Not aligned for trend following")
            
            # Check volatility conditions
            if 'conditions' in market_state and 'volatility' in market_state['conditions']:
                vol_regime = market_state['conditions']['volatility'].get('regime', 'UNKNOWN')
                if vol_regime == 'LOW':
                    lines.append(f"   {self._colorize('⚠️', 'warning')} Volatility: Too low for signal generation")
        
        lines.append("   💡 Possible reasons: Insufficient trend strength, misaligned indicators, or market filters")
        
        return lines
    
    def _generate_decision_path_section(self, decision_path: List[Dict[str, str]]) -> List[str]:
        """Generate decision path section"""
        lines = []
        
        lines.append(self._colorize("🧭 Decision Path:", 'info'))
        lines.append("-" * 40)
        
        for i, step in enumerate(decision_path, 1):
            status = step.get('status', 'UNKNOWN')
            description = step.get('description', 'Unknown step')
            
            if status == 'PASS':
                icon = self._colorize(self.symbols['check'], 'success')
            elif status == 'FAIL':
                icon = self._colorize(self.symbols['cross'], 'error')
            else:
                icon = self._colorize(self.symbols['warning'], 'warning')
            
            lines.append(f"   {i}. {icon} {description}")
        
        return lines
    
    # Removed misleading comparison and final result sections
    
    def _generate_footer(self, result: ValidationResult) -> List[str]:
        """Generate report footer"""
        lines = []
        
        lines.append("")
        lines.append("-" * self.max_line_length)
        lines.append(f"⏱️  Processing time: {result.processing_time_ms:.1f}ms")
        lines.append(f"📝 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return lines
    
    def _generate_batch_statistics(self, results: List[ValidationResult]) -> List[str]:
        """Generate batch statistics"""
        lines = []
        
        total = len(results)
        successful = sum(1 for r in results if r.success)
        signals_detected = sum(1 for r in results if r.signal_detected)
        with_stored_alerts = sum(1 for r in results if r.comparison_result and r.comparison_result.get('stored_alert_found'))
        
        lines.append(self._colorize("📈 Statistics:", 'info'))
        lines.append(f"   Successful Validations: {successful}/{total} ({successful/total*100:.1f}%)")
        lines.append(f"   Signals Detected: {signals_detected}/{total} ({signals_detected/total*100:.1f}%)")
        lines.append(f"   With Stored Alerts: {with_stored_alerts}/{total} ({with_stored_alerts/total*100:.1f}%)")
        
        # Average processing time
        avg_processing = sum(r.processing_time_ms for r in results) / total if total > 0 else 0
        lines.append(f"   Average Processing Time: {avg_processing:.1f}ms")
        
        return lines
    
    def _generate_epic_summary(self, results: List[ValidationResult]) -> List[str]:
        """Generate summary by epic"""
        lines = []
        
        epic_stats = {}
        for result in results:
            epic = result.epic
            if epic not in epic_stats:
                epic_stats[epic] = {'total': 0, 'successful': 0, 'signals': 0}
            
            epic_stats[epic]['total'] += 1
            if result.success:
                epic_stats[epic]['successful'] += 1
            if result.signal_detected:
                epic_stats[epic]['signals'] += 1
        
        lines.append(self._colorize("📊 Results by Epic:", 'info'))
        for epic, stats in epic_stats.items():
            success_rate = stats['successful'] / stats['total'] * 100
            signal_rate = stats['signals'] / stats['total'] * 100
            lines.append(f"   {epic}: {stats['successful']}/{stats['total']} success ({success_rate:.1f}%), {stats['signals']} signals ({signal_rate:.1f}%)")
        
        return lines
    
    def _generate_failure_summary(self, failed_results: List[ValidationResult]) -> List[str]:
        """Generate summary of failed validations"""
        lines = []
        
        lines.append(self._colorize("❌ Failed Validations:", 'error'))
        for result in failed_results[:5]:  # Show first 5 failures
            lines.append(f"   {result.epic} @ {result.timestamp.strftime('%H:%M:%S')}: {result.error_message}")
        
        if len(failed_results) > 5:
            lines.append(f"   ... and {len(failed_results) - 5} more failures")
        
        return lines
    
    def _generate_performance_summary(self, results: List[ValidationResult]) -> List[str]:
        """Generate performance summary"""
        lines = []
        
        processing_times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]
        
        if processing_times:
            lines.append(self._colorize("⚡ Performance Summary:", 'info'))
            lines.append(f"   Fastest: {min(processing_times):.1f}ms")
            lines.append(f"   Slowest: {max(processing_times):.1f}ms")
            lines.append(f"   Average: {sum(processing_times)/len(processing_times):.1f}ms")
        
        return lines
    
    def _generate_ema_analysis(self, signal: Dict[str, Any], show_calculations: bool) -> List[str]:
        """Generate EMA-specific analysis"""
        lines = []
        
        # EMA crossover analysis
        if show_calculations:
            lines.append("   📊 EMA Analysis:")
            
            # Check for EMA values in signal
            ema_fields = ['ema_21', 'ema_50', 'ema_200', 'ema_short', 'ema_long', 'ema_trend']
            ema_values = {k: v for k, v in signal.items() if k in ema_fields and v is not None}
            
            if ema_values:
                for ema, value in ema_values.items():
                    lines.append(f"     {ema.upper()}: {self._format_price(value)}")
            
            # Check trend alignment
            if signal.get('trend_aligned'):
                lines.append(f"     {self._colorize('✓', 'success')} Trend Alignment: Confirmed")
            
            # Momentum confirmation
            if 'momentum_confirmed' in signal:
                status = self._colorize('✓', 'success') if signal['momentum_confirmed'] else self._colorize('✗', 'error')
                lines.append(f"     {status} Momentum: {'Confirmed' if signal.get('momentum_confirmed') else 'Not confirmed'}")
        
        return lines
    
    def _generate_macd_analysis(self, signal: Dict[str, Any], show_calculations: bool) -> List[str]:
        """Generate MACD-specific analysis"""
        lines = []
        
        if show_calculations:
            lines.append("   📈 MACD Analysis:")
            
            macd_fields = ['macd', 'macd_signal', 'macd_histogram']
            for field in macd_fields:
                if field in signal:
                    lines.append(f"     {field.upper()}: {self._format_indicator(signal[field])}")
            
            # MACD crossover
            if 'macd_bullish_crossover' in signal:
                status = self._colorize('✓', 'success') if signal['macd_bullish_crossover'] else self._colorize('✗', 'error')
                lines.append(f"     {status} Bullish Crossover: {'Yes' if signal.get('macd_bullish_crossover') else 'No'}")
        
        return lines
    
    def _generate_kama_analysis(self, signal: Dict[str, Any], show_calculations: bool) -> List[str]:
        """Generate KAMA-specific analysis"""
        lines = []
        
        if show_calculations:
            lines.append("   🌊 KAMA Analysis:")
            
            if 'kama' in signal:
                lines.append(f"     KAMA: {self._format_price(signal['kama'])}")
            if 'efficiency_ratio' in signal:
                lines.append(f"     Efficiency Ratio: {self._format_indicator(signal['efficiency_ratio'])}")
            if 'market_regime' in signal:
                lines.append(f"     Market Regime: {signal['market_regime']}")
        
        return lines
    
    def _generate_filter_analysis(self, signal: Dict[str, Any]) -> List[str]:
        """Generate filter analysis"""
        lines = []
        
        filters = []
        
        # Common filters
        if 'large_candle_filtered' in signal:
            status = 'PASS' if not signal['large_candle_filtered'] else 'FAIL'
            filters.append(f"Large Candle Filter: {status}")
        
        if 'volume_filtered' in signal:
            status = 'PASS' if not signal['volume_filtered'] else 'FAIL'
            filters.append(f"Volume Filter: {status}")
        
        if 'confidence_filtered' in signal:
            status = 'PASS' if not signal['confidence_filtered'] else 'FAIL'
            filters.append(f"Confidence Filter: {status}")
        
        if filters:
            lines.append("   🔍 Filters:")
            for filter_result in filters:
                status = filter_result.split(': ')[1]
                icon = self._colorize('✓', 'success') if status == 'PASS' else self._colorize('✗', 'error')
                lines.append(f"     {icon} {filter_result}")
        
        return lines
    
    def _generate_smart_money_analysis(self, signal: Dict[str, Any]) -> List[str]:
        """Generate Smart Money analysis"""
        lines = []
        
        lines.append("   🧠 Smart Money Analysis:")
        lines.append(f"     Validated: {self._colorize('✓', 'success')}")
        
        if 'smart_money_type' in signal:
            lines.append(f"     Type: {signal['smart_money_type']}")
        
        if 'smart_money_score' in signal:
            score = signal['smart_money_score']
            lines.append(f"     Score: {self._format_indicator(score)}")
        
        if 'enhanced_confidence_score' in signal:
            original = signal.get('confidence_score', 0)
            enhanced = signal['enhanced_confidence_score']
            lines.append(f"     Enhanced Confidence: {self._format_percentage(original)} → {self._format_percentage(enhanced)}")
        
        return lines
    
    def _format_price(self, price: float) -> str:
        """Format price value"""
        try:
            # Ensure price is numeric
            if isinstance(price, str):
                price = float(price)
            if abs(price) < 1:
                return f"{price:.{self.decimal_places['price']}f}"
            else:
                return f"{price:.{max(2, self.decimal_places['price']-2)}f}"
        except (ValueError, TypeError):
            return str(price)
    
    def _format_percentage(self, value: float) -> str:
        """Format percentage value"""
        try:
            # Ensure value is numeric
            if isinstance(value, str):
                value = float(value)
            decimal_places = self.decimal_places['percentage']
            return f"{value:.{decimal_places}f}%"
        except (ValueError, TypeError):
            return f"{value}%"
    
    def _format_indicator(self, value: float) -> str:
        """Format indicator value"""
        try:
            # Ensure value is numeric
            if isinstance(value, str):
                value = float(value)
            return f"{value:.{self.decimal_places['indicator']}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text"""
        if not self.use_colors:
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"