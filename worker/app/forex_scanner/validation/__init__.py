# validation/__init__.py
"""
Comprehensive Signal Validation and Statistical Analysis Module

This module provides comprehensive signal validation capabilities with two main systems:

1. Signal Replay and Validation:
   - Replay historical scanning conditions and validate alert generation
   - Recreation of historical scanner states
   - Detailed validation reporting

2. Statistical Validation Framework:
   - Comprehensive statistical validation for trading strategy backtests
   - Real-time performance correlation analysis
   - Advanced overfitting detection with multiple methods
   - Statistical significance testing with hypothesis testing
   - Historical data quality assurance
   - Backtest/live pipeline consistency validation
   - Automated validation reports with statistical confidence scoring
   - Continuous real-time monitoring capabilities

Components:

Signal Replay System:
- HistoricalDataManager: Historical data retrieval and management
- ScannerStateRecreator: Recreation of historical scanner states
- ReplayEngine: Core orchestration of the replay process
- ValidationReporter: Detailed validation reporting
- SignalReplayValidator: Main entry point for validation operations

Statistical Validation System:
- StatisticalValidationFramework: Core validation orchestrator
- RealtimeCorrelationAnalyzer: Real-time performance correlation analysis
- OverfittingDetector: Advanced overfitting detection with multiple methods
- StatisticalSignificanceTester: Comprehensive hypothesis testing
- DataQualityValidator: Historical data quality assurance
- PipelineConsistencyValidator: Backtest/live pipeline consistency validation
- ValidationReportGenerator: Automated validation reports
- RealtimeValidationMonitor: Continuous monitoring capabilities

Usage:

Signal Replay:
    python -m forex_scanner.validation.signal_replay_validator \
        --timestamp "2025-01-15 14:30:00" \
        --epic "CS.D.EURUSD.MINI.IP"

Statistical Validation:
    from forex_scanner.validation import create_statistical_validation_framework

    framework = create_statistical_validation_framework(db_manager)
    results = framework.validate_backtest_execution(execution_id)
"""

__version__ = "2.0.0"
__author__ = "TradeSystemV1 Team"

# Signal Replay System (existing)
from .historical_data_manager import HistoricalDataManager
from .scanner_state_recreator import ScannerStateRecreator
from .replay_engine import ReplayEngine
from .validation_reporter import ValidationReporter
from .signal_replay_validator import SignalReplayValidator

# Statistical Validation Framework (new)
from .statistical_validation_framework import (
    StatisticalValidationFramework,
    ValidationLevel,
    TestResult,
    ValidationResult,
    create_statistical_validation_framework
)

from .realtime_correlation_analyzer import (
    RealtimeCorrelationAnalyzer,
    CorrelationMetrics,
    RegimeCorrelation,
    create_realtime_correlation_analyzer
)

from .overfitting_detector import (
    OverfittingDetector,
    OverfittingRisk,
    OverfittingAssessment,
    WalkForwardResults,
    CrossValidationResults,
    create_overfitting_detector
)

from .statistical_significance_tester import (
    StatisticalSignificanceTester,
    TestType,
    SignificanceLevel,
    StatisticalTest,
    MultipleTestingResults,
    create_statistical_significance_tester
)

from .data_quality_validator import (
    DataQualityValidator,
    DataQualityLevel,
    QualityCheck,
    QualityMetric,
    create_data_quality_validator
)

from .pipeline_consistency_validator import (
    PipelineConsistencyValidator,
    ConsistencyLevel,
    ValidationComponent,
    ConsistencyTest,
    create_pipeline_consistency_validator
)

from .validation_report_generator import (
    ValidationReportGenerator,
    ReportType,
    ConfidenceLevel as ReportConfidenceLevel,
    ValidationSummary,
    ReportSection,
    create_validation_report_generator
)

from .realtime_validation_monitor import (
    RealtimeValidationMonitor,
    AlertSeverity,
    MonitoringMetric,
    Alert,
    ValidationStatus,
    create_realtime_validation_monitor
)

__all__ = [
    # Signal Replay System (existing)
    'HistoricalDataManager',
    'ScannerStateRecreator',
    'ReplayEngine',
    'ValidationReporter',
    'SignalReplayValidator',

    # Statistical Validation Framework (new)
    # Core Framework
    'StatisticalValidationFramework',
    'ValidationLevel',
    'TestResult',
    'ValidationResult',
    'create_statistical_validation_framework',

    # Real-time Correlation
    'RealtimeCorrelationAnalyzer',
    'CorrelationMetrics',
    'RegimeCorrelation',
    'create_realtime_correlation_analyzer',

    # Overfitting Detection
    'OverfittingDetector',
    'OverfittingRisk',
    'OverfittingAssessment',
    'WalkForwardResults',
    'CrossValidationResults',
    'create_overfitting_detector',

    # Statistical Significance
    'StatisticalSignificanceTester',
    'TestType',
    'SignificanceLevel',
    'StatisticalTest',
    'MultipleTestingResults',
    'create_statistical_significance_tester',

    # Data Quality
    'DataQualityValidator',
    'DataQualityLevel',
    'QualityCheck',
    'QualityMetric',
    'create_data_quality_validator',

    # Pipeline Consistency
    'PipelineConsistencyValidator',
    'ConsistencyLevel',
    'ValidationComponent',
    'ConsistencyTest',
    'create_pipeline_consistency_validator',

    # Report Generation
    'ValidationReportGenerator',
    'ReportType',
    'ReportConfidenceLevel',
    'ValidationSummary',
    'ReportSection',
    'create_validation_report_generator',

    # Real-time Monitoring
    'RealtimeValidationMonitor',
    'AlertSeverity',
    'MonitoringMetric',
    'Alert',
    'ValidationStatus',
    'create_realtime_validation_monitor',
]


def create_complete_validation_system(db_manager):
    """
    Create a complete statistical validation system with all components integrated

    Args:
        db_manager: DatabaseManager instance

    Returns:
        Dictionary containing all statistical validation components
    """

    # Create core framework
    statistical_framework = create_statistical_validation_framework(db_manager)

    # Create individual components
    correlation_analyzer = create_realtime_correlation_analyzer(db_manager)
    overfitting_detector = create_overfitting_detector(db_manager)
    significance_tester = create_statistical_significance_tester(db_manager)
    quality_validator = create_data_quality_validator(db_manager)
    consistency_validator = create_pipeline_consistency_validator(db_manager)

    # Create report generator
    report_generator = create_validation_report_generator(db_manager, statistical_framework)

    # Create real-time monitor
    realtime_monitor = create_realtime_validation_monitor(
        db_manager, statistical_framework, report_generator
    )

    return {
        'statistical_framework': statistical_framework,
        'correlation_analyzer': correlation_analyzer,
        'overfitting_detector': overfitting_detector,
        'significance_tester': significance_tester,
        'quality_validator': quality_validator,
        'consistency_validator': consistency_validator,
        'report_generator': report_generator,
        'realtime_monitor': realtime_monitor
    }


def validate_execution(db_manager, execution_id: int, generate_report: bool = True):
    """
    Convenience function to quickly validate a backtest execution

    Args:
        db_manager: DatabaseManager instance
        execution_id: Backtest execution ID to validate
        generate_report: Whether to generate a comprehensive report

    Returns:
        Validation results and optional report
    """

    framework = create_statistical_validation_framework(db_manager)
    results = framework.validate_backtest_execution(execution_id)

    if generate_report:
        report_generator = create_validation_report_generator(db_manager, framework)
        report = report_generator.generate_comprehensive_report(execution_id)
        return results, report

    return results, None