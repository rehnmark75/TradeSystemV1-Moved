"""
Data models for TradingView Scripts MCP Server

Defines data structures and validation for script metadata,
Pine Script analysis results, and strategy configurations.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json

@dataclass
class ScriptMetadata:
    """TradingView script metadata"""
    slug: str
    title: str
    author: str
    tags: str
    open_source: bool
    url: str
    description: Optional[str] = None
    likes_count: int = 0
    uses_count: int = 0
    script_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'slug': self.slug,
            'title': self.title,
            'author': self.author,
            'tags': self.tags,
            'open_source': self.open_source,
            'url': self.url,
            'description': self.description,
            'likes_count': self.likes_count,
            'uses_count': self.uses_count,
            'script_type': self.script_type
        }

@dataclass
class PineInput:
    """Pine Script input parameter"""
    type: str  # int, float, bool, string
    label: str
    default: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'type': self.type,
            'label': self.label,
            'default': self.default
        }

@dataclass
class PineSignals:
    """Extracted Pine Script signals and patterns"""
    ema_periods: List[int]
    has_cross_up: bool
    has_cross_down: bool
    macd: Optional[Dict[str, int]]
    higher_tf: List[Dict[str, str]]
    mentions_fvg: bool
    mentions_smc: bool
    rsi_periods: List[int]
    bollinger_bands: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ema_periods': sorted(self.ema_periods),
            'has_cross_up': self.has_cross_up,
            'has_cross_down': self.has_cross_down,
            'macd': self.macd,
            'higher_tf': self.higher_tf,
            'mentions_fvg': self.mentions_fvg,
            'mentions_smc': self.mentions_smc,
            'rsi_periods': sorted(self.rsi_periods),
            'bollinger_bands': self.bollinger_bands
        }

@dataclass
class StrategyConfig:
    """Generated strategy configuration for TradeSystemV1"""
    name: str
    provenance: Dict[str, Any]
    modules: Dict[str, Any]
    filters: Dict[str, Any]
    rules: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'provenance': self.provenance,
            'modules': self.modules,
            'filters': self.filters,
            'rules': self.rules
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class AnalysisResult:
    """Pine Script analysis result"""
    inputs: List[PineInput]
    signals: PineSignals
    complexity_score: float
    confidence_score: float
    strategy_type: str  # trending, ranging, breakout, scalping
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'inputs': [inp.to_dict() for inp in self.inputs],
            'signals': self.signals.to_dict(),
            'complexity_score': self.complexity_score,
            'confidence_score': self.confidence_score,
            'strategy_type': self.strategy_type
        }

@dataclass
class ImportResult:
    """Strategy import result"""
    success: bool
    config: Optional[StrategyConfig]
    analysis: Optional[AnalysisResult]
    error: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'config': self.config.to_dict() if self.config else None,
            'analysis': self.analysis.to_dict() if self.analysis else None,
            'error': self.error
        }