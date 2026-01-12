"""
Deep Analysis Module

Provides deep analysis capabilities for stock scanner signals.

Components:
- DeepAnalysisOrchestrator: Main coordinator
- TechnicalDeepAnalyzer: MTF, volume, SMC analysis
- FundamentalDeepAnalyzer: Quality, catalyst, institutional analysis
- ContextualDeepAnalyzer: News, regime, sector analysis

Usage:
    from stock_scanner.services.deep_analysis import DeepAnalysisOrchestrator

    orchestrator = DeepAnalysisOrchestrator(db_manager)
    result = await orchestrator.analyze_signal(signal_id)
    print(f"DAQ Score: {result.daq_score} ({result.daq_grade.value})")
"""

from .models import (
    # Result types
    DeepAnalysisResult,
    TechnicalDeepResult,
    FundamentalDeepResult,
    ContextualDeepResult,
    # Component results
    MTFAnalysisResult,
    VolumeAnalysisResult,
    SMCAnalysisResult,
    QualityScreenResult,
    CatalystAnalysisResult,
    InstitutionalAnalysisResult,
    NewsSentimentResult,
    MarketRegimeResult,
    SectorRotationResult,
    TimeframeAnalysis,
    # Enums
    DAQGrade,
    TrendDirection,
    MarketRegime,
    # Config
    DeepAnalysisConfig,
)

from .orchestrator import DeepAnalysisOrchestrator
from .technical_analyzer import TechnicalDeepAnalyzer
from .fundamental_analyzer import FundamentalDeepAnalyzer
from .contextual_analyzer import ContextualDeepAnalyzer

__all__ = [
    # Main orchestrator
    'DeepAnalysisOrchestrator',
    # Analyzers
    'TechnicalDeepAnalyzer',
    'FundamentalDeepAnalyzer',
    'ContextualDeepAnalyzer',
    # Result types
    'DeepAnalysisResult',
    'TechnicalDeepResult',
    'FundamentalDeepResult',
    'ContextualDeepResult',
    'MTFAnalysisResult',
    'VolumeAnalysisResult',
    'SMCAnalysisResult',
    'QualityScreenResult',
    'CatalystAnalysisResult',
    'InstitutionalAnalysisResult',
    'NewsSentimentResult',
    'MarketRegimeResult',
    'SectorRotationResult',
    'TimeframeAnalysis',
    # Enums
    'DAQGrade',
    'TrendDirection',
    'MarketRegime',
    # Config
    'DeepAnalysisConfig',
]
