"""
Signal Intelligence Service - Comprehensive trading signal intelligence platform
Combines log parsing, database correlation, and analytics for actionable insights
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from enhanced_log_parser import (
    EnhancedLogParser, ParsedLogEntry, SignalContext,
    LogSource, AlertType, LogLevel
)
from log_database_correlator import (
    LogDatabaseCorrelator, EnrichedSignalEntry, SignalPerformanceMetrics
)

logger = logging.getLogger(__name__)

class TimeRange(Enum):
    LAST_HOUR = "1h"
    LAST_4_HOURS = "4h"
    LAST_24_HOURS = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"

class SignalQuality(Enum):
    EXCELLENT = "excellent"  # >80% confidence, historical success
    GOOD = "good"           # 60-80% confidence
    MODERATE = "moderate"    # 40-60% confidence
    POOR = "poor"           # <40% confidence
    REJECTED = "rejected"    # Signal was rejected

@dataclass
class SignalSummary:
    """High-level signal activity summary"""
    total_signals: int = 0
    signals_detected: int = 0
    signals_rejected: int = 0
    avg_confidence: float = 0.0
    top_strategy: Optional[str] = None
    top_epic: Optional[str] = None
    last_signal_time: Optional[datetime] = None
    success_rate_estimate: float = 0.0

@dataclass
class SystemHealthSummary:
    """System health and performance summary"""
    overall_status: str = "unknown"  # healthy, warning, critical
    forex_scanner_status: str = "unknown"
    stream_health_status: str = "unknown"
    last_error: Optional[Dict[str, Any]] = None
    last_warning: Optional[Dict[str, Any]] = None
    error_count_24h: int = 0
    warning_count_24h: int = 0
    uptime_indicators: Dict[str, str] = None

@dataclass
class AnalyticsData:
    """Analytics and visualization data"""
    signal_trends: Dict[str, Any] = None
    strategy_performance: Dict[str, Any] = None
    epic_analysis: Dict[str, Any] = None
    hourly_activity: Dict[str, Any] = None
    rejection_analysis: Dict[str, Any] = None

class SignalIntelligenceService:
    """
    Comprehensive signal intelligence service providing:
    - Real-time signal monitoring and analysis
    - Historical performance correlation
    - System health monitoring
    - Advanced analytics and insights
    """

    def __init__(self, database_url: str = None):
        """Initialize signal intelligence service"""
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.log_parser = EnhancedLogParser()
        self.db_correlator = LogDatabaseCorrelator(database_url)

        # Cache for performance
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 60  # 1 minute cache TTL

        self.logger.info("✅ Signal Intelligence Service initialized")

    def get_real_time_signal_feed(self, max_entries: int = 20) -> List[EnrichedSignalEntry]:
        """Get real-time feed of enriched signal intelligence"""
        try:
            # Get enriched signals from last hour
            signals = self.db_correlator.get_enriched_signal_intelligence(
                hours_back=1,
                max_signals=max_entries
            )

            return signals

        except Exception as e:
            self.logger.error(f"Error getting real-time signal feed: {e}")
            return []

    def get_signal_summary(self, time_range: TimeRange = TimeRange.LAST_24_HOURS) -> SignalSummary:
        """Get high-level signal activity summary"""
        cache_key = f"signal_summary_{time_range.value}"

        # Check cache
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            hours_map = {
                TimeRange.LAST_HOUR: 1,
                TimeRange.LAST_4_HOURS: 4,
                TimeRange.LAST_24_HOURS: 24,
                TimeRange.LAST_WEEK: 168,
                TimeRange.LAST_MONTH: 720
            }

            hours_back = hours_map.get(time_range, 24)

            # Get signal logs
            signal_logs = self.log_parser.get_signal_intelligence(
                hours_back=hours_back,
                max_signals=200
            )

            # Analyze signals
            summary = SignalSummary()
            summary.total_signals = len(signal_logs)

            detected_signals = []
            rejected_signals = []
            confidences = []
            strategies = []
            epics = []

            for log in signal_logs:
                if log.alert_type == AlertType.SIGNAL_DETECTED.value:
                    detected_signals.append(log)
                    if log.signal_context and log.signal_context.confidence:
                        confidences.append(log.signal_context.confidence)
                elif log.alert_type == AlertType.SIGNAL_REJECTED.value:
                    rejected_signals.append(log)

                if log.signal_context:
                    if log.signal_context.strategy:
                        strategies.append(log.signal_context.strategy)

                if log.epic:
                    epics.append(log.epic)

            summary.signals_detected = len(detected_signals)
            summary.signals_rejected = len(rejected_signals)

            if confidences:
                summary.avg_confidence = np.mean(confidences)

            if strategies:
                summary.top_strategy = Counter(strategies).most_common(1)[0][0]

            if epics:
                summary.top_epic = Counter(epics).most_common(1)[0][0]

            if signal_logs:
                summary.last_signal_time = max(log.timestamp for log in signal_logs)

            # Estimate success rate based on confidence and historical data
            if confidences:
                summary.success_rate_estimate = np.mean(confidences) * 0.8  # Conservative estimate

            # Cache result
            self._cache[cache_key] = summary
            self._cache_timestamps[cache_key] = datetime.now()

            return summary

        except Exception as e:
            self.logger.error(f"Error getting signal summary: {e}")
            return SignalSummary()

    def get_system_health(self) -> SystemHealthSummary:
        """Get comprehensive system health summary"""
        cache_key = "system_health"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            # Get health data from log parser
            health_data = self.log_parser.get_system_health_summary(hours_back=24)

            summary = SystemHealthSummary()

            # Determine overall status
            error_count = health_data.get('by_level', {}).get('ERROR', 0)
            warning_count = health_data.get('by_level', {}).get('WARNING', 0)

            summary.error_count_24h = error_count
            summary.warning_count_24h = warning_count

            if error_count > 10:
                summary.overall_status = "critical"
            elif error_count > 0 or warning_count > 20:
                summary.overall_status = "warning"
            else:
                summary.overall_status = "healthy"

            # Extract specific service health
            summary.forex_scanner_status = health_data.get('forex_scanner_health', 'unknown')
            summary.stream_health_status = health_data.get('stream_health', 'unknown')

            # Last error and warning
            if health_data.get('last_error'):
                error_log = health_data['last_error']
                summary.last_error = {
                    'timestamp': error_log.timestamp.isoformat(),
                    'message': error_log.message,
                    'module': error_log.module
                }

            if health_data.get('last_warning'):
                warning_log = health_data['last_warning']
                summary.last_warning = {
                    'timestamp': warning_log.timestamp.isoformat(),
                    'message': warning_log.message,
                    'module': warning_log.module
                }

            # Uptime indicators
            summary.uptime_indicators = {
                'forex_scanner': summary.forex_scanner_status,
                'stream_service': summary.stream_health_status,
                'database': 'healthy' if error_count < 5 else 'issues'
            }

            # Cache result
            self._cache[cache_key] = summary
            self._cache_timestamps[cache_key] = datetime.now()

            return summary

        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return SystemHealthSummary()

    def get_analytics_data(self, time_range: TimeRange = TimeRange.LAST_24_HOURS) -> AnalyticsData:
        """Get comprehensive analytics data for visualizations"""
        cache_key = f"analytics_{time_range.value}"

        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            hours_map = {
                TimeRange.LAST_HOUR: 1,
                TimeRange.LAST_4_HOURS: 4,
                TimeRange.LAST_24_HOURS: 24,
                TimeRange.LAST_WEEK: 168,
                TimeRange.LAST_MONTH: 720
            }

            hours_back = hours_map.get(time_range, 24)

            # Get comprehensive log data
            signal_logs = self.log_parser.get_signal_intelligence(
                hours_back=hours_back,
                max_signals=500
            )

            analytics = AnalyticsData()

            # Signal trends over time
            analytics.signal_trends = self._analyze_signal_trends(signal_logs)

            # Strategy performance
            analytics.strategy_performance = self._analyze_strategy_performance(signal_logs)

            # Epic analysis
            analytics.epic_analysis = self._analyze_epic_performance(signal_logs)

            # Hourly activity
            analytics.hourly_activity = self._analyze_hourly_activity(signal_logs)

            # Rejection analysis
            analytics.rejection_analysis = self._analyze_rejection_patterns(signal_logs)

            # Cache result
            self._cache[cache_key] = analytics
            self._cache_timestamps[cache_key] = datetime.now()

            return analytics

        except Exception as e:
            self.logger.error(f"Error getting analytics data: {e}")
            return AnalyticsData()

    def search_signals(self,
                      query: str = None,
                      strategy: str = None,
                      epic: str = None,
                      signal_type: str = None,
                      log_level: str = None,
                      time_range: TimeRange = TimeRange.LAST_24_HOURS,
                      max_results: int = 100) -> List[ParsedLogEntry]:
        """Advanced signal search with multiple criteria"""
        try:
            hours_map = {
                TimeRange.LAST_HOUR: 1,
                TimeRange.LAST_4_HOURS: 4,
                TimeRange.LAST_24_HOURS: 24,
                TimeRange.LAST_WEEK: 168,
                TimeRange.LAST_MONTH: 720
            }

            hours_back = hours_map.get(time_range, 24)

            # Get all relevant logs
            all_logs = self.log_parser.get_recent_logs(
                source=LogSource.FOREX_SCANNER,
                hours_back=hours_back,
                max_entries=1000
            )

            # Apply filters
            filtered_logs = []

            for log in all_logs:
                # Text search in message
                if query and query.lower() not in log.message.lower():
                    continue

                # Strategy filter
                if (strategy and
                    log.signal_context and
                    log.signal_context.strategy and
                    log.signal_context.strategy.lower() != strategy.lower()):
                    continue

                # Epic filter
                if epic and log.epic and log.epic.lower() != epic.lower():
                    continue

                # Signal type filter
                if (signal_type and
                    log.signal_context and
                    log.signal_context.signal_type and
                    log.signal_context.signal_type.lower() != signal_type.lower()):
                    continue

                # Log level filter
                if log_level and log.level.lower() != log_level.lower():
                    continue

                filtered_logs.append(log)

            return filtered_logs[:max_results]

        except Exception as e:
            self.logger.error(f"Error searching signals: {e}")
            return []

    def get_signal_quality_distribution(self, time_range: TimeRange = TimeRange.LAST_24_HOURS) -> Dict[str, int]:
        """Get distribution of signal quality levels"""
        try:
            hours_map = {
                TimeRange.LAST_HOUR: 1,
                TimeRange.LAST_4_HOURS: 4,
                TimeRange.LAST_24_HOURS: 24,
                TimeRange.LAST_WEEK: 168,
                TimeRange.LAST_MONTH: 720
            }

            hours_back = hours_map.get(time_range, 24)
            signal_logs = self.log_parser.get_signal_intelligence(hours_back=hours_back)

            quality_counts = {
                SignalQuality.EXCELLENT.value: 0,
                SignalQuality.GOOD.value: 0,
                SignalQuality.MODERATE.value: 0,
                SignalQuality.POOR.value: 0,
                SignalQuality.REJECTED.value: 0
            }

            for log in signal_logs:
                if log.alert_type == AlertType.SIGNAL_REJECTED.value:
                    quality_counts[SignalQuality.REJECTED.value] += 1
                elif log.signal_context and log.signal_context.confidence:
                    conf = log.signal_context.confidence
                    if conf >= 0.8:
                        quality_counts[SignalQuality.EXCELLENT.value] += 1
                    elif conf >= 0.6:
                        quality_counts[SignalQuality.GOOD.value] += 1
                    elif conf >= 0.4:
                        quality_counts[SignalQuality.MODERATE.value] += 1
                    else:
                        quality_counts[SignalQuality.POOR.value] += 1

            return quality_counts

        except Exception as e:
            self.logger.error(f"Error getting signal quality distribution: {e}")
            return {}

    def _analyze_signal_trends(self, signal_logs: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Analyze signal trends over time"""
        hourly_counts = defaultdict(int)
        hourly_confidence = defaultdict(list)

        for log in signal_logs:
            if log.alert_type == AlertType.SIGNAL_DETECTED.value:
                hour_key = log.timestamp.strftime('%H:00')
                hourly_counts[hour_key] += 1

                if log.signal_context and log.signal_context.confidence:
                    hourly_confidence[hour_key].append(log.signal_context.confidence)

        # Calculate average confidence per hour
        avg_confidence = {}
        for hour, confidences in hourly_confidence.items():
            avg_confidence[hour] = np.mean(confidences) if confidences else 0

        return {
            'hourly_counts': dict(hourly_counts),
            'hourly_avg_confidence': avg_confidence,
            'total_signals': sum(hourly_counts.values())
        }

    def _analyze_strategy_performance(self, signal_logs: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        strategy_stats = defaultdict(lambda: {
            'detected': 0,
            'rejected': 0,
            'avg_confidence': 0,
            'confidences': []
        })

        for log in signal_logs:
            if log.signal_context and log.signal_context.strategy:
                strategy = log.signal_context.strategy

                if log.alert_type == AlertType.SIGNAL_DETECTED.value:
                    strategy_stats[strategy]['detected'] += 1
                    if log.signal_context.confidence:
                        strategy_stats[strategy]['confidences'].append(log.signal_context.confidence)
                elif log.alert_type == AlertType.SIGNAL_REJECTED.value:
                    strategy_stats[strategy]['rejected'] += 1

        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = np.mean(stats['confidences'])
                del stats['confidences']  # Remove raw data

        return dict(strategy_stats)

    def _analyze_epic_performance(self, signal_logs: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Analyze performance by currency pair"""
        epic_stats = defaultdict(lambda: {
            'signal_count': 0,
            'avg_confidence': 0,
            'confidences': []
        })

        for log in signal_logs:
            if log.epic and log.alert_type == AlertType.SIGNAL_DETECTED.value:
                epic_stats[log.epic]['signal_count'] += 1
                if log.signal_context and log.signal_context.confidence:
                    epic_stats[log.epic]['confidences'].append(log.signal_context.confidence)

        # Calculate averages
        for epic, stats in epic_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = np.mean(stats['confidences'])
                del stats['confidences']  # Remove raw data

        return dict(epic_stats)

    def _analyze_hourly_activity(self, signal_logs: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Analyze activity patterns by hour"""
        hourly_activity = defaultdict(lambda: {
            'total_entries': 0,
            'signals': 0,
            'rejections': 0,
            'errors': 0,
            'warnings': 0
        })

        for log in signal_logs:
            hour = log.timestamp.hour
            hourly_activity[hour]['total_entries'] += 1

            if log.alert_type == AlertType.SIGNAL_DETECTED.value:
                hourly_activity[hour]['signals'] += 1
            elif log.alert_type == AlertType.SIGNAL_REJECTED.value:
                hourly_activity[hour]['rejections'] += 1

            if log.level == 'ERROR':
                hourly_activity[hour]['errors'] += 1
            elif log.level == 'WARNING':
                hourly_activity[hour]['warnings'] += 1

        return dict(hourly_activity)

    def _analyze_rejection_patterns(self, signal_logs: List[ParsedLogEntry]) -> Dict[str, Any]:
        """Analyze signal rejection patterns"""
        rejection_reasons = defaultdict(int)
        rejection_by_strategy = defaultdict(int)

        for log in signal_logs:
            if log.alert_type == AlertType.SIGNAL_REJECTED.value:
                if log.signal_context and log.signal_context.rejection_reason:
                    reason = log.signal_context.rejection_reason[:50]  # Truncate long reasons
                    rejection_reasons[reason] += 1

                if log.signal_context and log.signal_context.strategy:
                    rejection_by_strategy[log.signal_context.strategy] += 1

        return {
            'rejection_reasons': dict(rejection_reasons),
            'rejection_by_strategy': dict(rejection_by_strategy),
            'total_rejections': sum(rejection_reasons.values())
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        return (datetime.now() - cache_time).total_seconds() < self._cache_ttl

def get_signal_intelligence_service() -> SignalIntelligenceService:
    """Get a configured signal intelligence service instance"""
    return SignalIntelligenceService()

if __name__ == "__main__":
    # Test the signal intelligence service
    service = SignalIntelligenceService()

    print("Testing Signal Intelligence Service...")

    # Test signal summary
    summary = service.get_signal_summary(TimeRange.LAST_24_HOURS)
    print(f"\nSignal Summary (24h):")
    print(f"  Total: {summary.total_signals}")
    print(f"  Detected: {summary.signals_detected}")
    print(f"  Rejected: {summary.signals_rejected}")
    print(f"  Avg Confidence: {summary.avg_confidence:.1%}")
    print(f"  Top Strategy: {summary.top_strategy}")

    # Test system health
    health = service.get_system_health()
    print(f"\nSystem Health:")
    print(f"  Overall: {health.overall_status}")
    print(f"  Errors (24h): {health.error_count_24h}")
    print(f"  Warnings (24h): {health.warning_count_24h}")

    # Test real-time feed
    feed = service.get_real_time_signal_feed(max_entries=5)
    print(f"\nReal-time Feed ({len(feed)} entries):")
    for entry in feed:
        log = entry.log_entry
        print(f"  {log.timestamp} - {log.alert_type} - {log.message[:50]}...")