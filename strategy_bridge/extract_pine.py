"""
Pine Script Pattern Extractor

Extracts indicators, parameters, signals, and patterns from Pine Script code.
Supports EMA, MACD, RSI, Bollinger Bands, Smart Money Concepts, and more.
"""

import re
import logging
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Comprehensive regex patterns for Pine Script analysis
INPUT_RE = re.compile(r'\binput\.(int|float|bool|string|color|timeframe)\(\s*([^,)]+)\s*,\s*["\']([^"\']+)["\']\s*(?:,\s*[^)]*)?', re.I)
EMA_RE = re.compile(r'\b(?:ta\.)?ema\(\s*([a-zA-Z0-9_.]+)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)', re.I)
SMA_RE = re.compile(r'\b(?:ta\.)?sma\(\s*([a-zA-Z0-9_.]+)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)', re.I)
MACD_RE = re.compile(r'\b(?:ta\.)?macd\(\s*([a-zA-Z0-9_.]+)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)', re.I)
RSI_RE = re.compile(r'\b(?:ta\.)?rsi\(\s*([a-zA-Z0-9_.]+)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)', re.I)
BB_RE = re.compile(r'\b(?:ta\.)?bb\(\s*([a-zA-Z0-9_.]+)\s*,\s*([0-9]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([0-9.]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)', re.I)

# Crossover patterns
XOVER_RE = re.compile(r'\b(?:ta\.)?crossover\(\s*([a-zA-Z0-9_.]+)\s*,\s*([a-zA-Z0-9_.]+)\s*\)', re.I)
XUNDER_RE = re.compile(r'\b(?:ta\.)?crossunder\(\s*([a-zA-Z0-9_.]+)\s*,\s*([a-zA-Z0-9_.]+)\s*\)', re.I)

# Higher timeframe analysis
HTF_RE = re.compile(r'\brequest\.security\(\s*([a-zA-Z0-9_"\']+)\s*,\s*([a-zA-Z0-9_"\']+)\s*,', re.I)

# Smart Money Concepts patterns
FVG_PATTERNS = [
    re.compile(r'\b(fvg|fair\s*value\s*gap|imbalance)\b', re.I),
    re.compile(r'\bgap\s*(fill|unfilled|mitigation)\b', re.I),
    re.compile(r'\bwick\s*(fill|imbalance)\b', re.I)
]

SMC_PATTERNS = [
    re.compile(r'\b(BOS|break\s*of\s*structure)\b', re.I),
    re.compile(r'\b(CHoCH|change\s*of\s*character)\b', re.I),
    re.compile(r'\b(order\s*block|orderblock)\b', re.I),
    re.compile(r'\b(liquidity\s*(pool|grab|sweep))\b', re.I),
    re.compile(r'\b(displacement|premium|discount)\b', re.I),
    re.compile(r'\b(institutional|smart\s*money)\b', re.I),
    re.compile(r'\b(pd\s*array|premium\s*discount)\b', re.I)
]

# Volume patterns
VOLUME_PATTERNS = [
    re.compile(r'\bvolume\s*>', re.I),
    re.compile(r'\bvolume\s*profile\b', re.I),
    re.compile(r'\bvwap\b', re.I),
    re.compile(r'\bmoney\s*flow\b', re.I)
]

# Support/Resistance patterns
SR_PATTERNS = [
    re.compile(r'\b(support|resistance)\b', re.I),
    re.compile(r'\b(pivot|level)\b', re.I),
    re.compile(r'\b(horizontal|trendline)\b', re.I)
]

@dataclass
class ExtractedInput:
    """Represents a Pine Script input parameter"""
    type: str
    default: str
    label: str
    variable_name: str = ""

@dataclass
class ExtractedSignals:
    """Represents extracted signals and patterns from Pine Script"""
    ema_periods: Set[int]
    sma_periods: Set[int]
    has_cross_up: bool
    has_cross_down: bool
    macd: Optional[Dict[str, int]]
    rsi_periods: Set[int]
    bollinger_bands: Optional[Dict[str, Any]]
    higher_tf: List[Dict[str, str]]
    mentions_fvg: bool
    mentions_smc: bool
    mentions_volume: bool
    mentions_sr: bool
    complexity_score: float
    strategy_type: str

def extract_inputs(pine_code: str) -> List[Dict[str, str]]:
    """
    Extract input parameters from Pine Script code
    
    Args:
        pine_code: Pine Script source code
        
    Returns:
        List of input parameter dictionaries
    """
    inputs = []
    
    try:
        for match in INPUT_RE.finditer(pine_code):
            input_type, default_value, label = match.groups()
            
            # Clean up values
            default_value = default_value.strip()
            label = label.strip()
            
            inputs.append({
                "type": input_type,
                "label": label,
                "default": default_value
            })
            
        logger.info(f"Extracted {len(inputs)} input parameters")
        return inputs
        
    except Exception as e:
        logger.error(f"Failed to extract inputs: {e}")
        return []

def extract_signals(pine_code: str) -> Dict[str, Any]:
    """
    Extract signals and patterns from Pine Script code
    
    Args:
        pine_code: Pine Script source code
        
    Returns:
        Dictionary containing extracted signals and patterns
    """
    try:
        signals = ExtractedSignals(
            ema_periods=set(),
            sma_periods=set(),
            has_cross_up=False,
            has_cross_down=False,
            macd=None,
            rsi_periods=set(),
            bollinger_bands=None,
            higher_tf=[],
            mentions_fvg=False,
            mentions_smc=False,
            mentions_volume=False,
            mentions_sr=False,
            complexity_score=0.0,
            strategy_type="unknown"
        )
        
        # Extract EMA periods
        for match in EMA_RE.finditer(pine_code):
            _, period = match.groups()
            try:
                if period.isdigit():
                    signals.ema_periods.add(int(period))
                else:
                    # Handle variable references - look for common values
                    if 'fast' in period.lower():
                        signals.ema_periods.add(12)  # Common fast EMA
                    elif 'slow' in period.lower():
                        signals.ema_periods.add(26)  # Common slow EMA
                    elif 'trend' in period.lower():
                        signals.ema_periods.add(200)  # Common trend EMA
            except ValueError:
                continue
        
        # Extract SMA periods
        for match in SMA_RE.finditer(pine_code):
            _, period = match.groups()
            try:
                if period.isdigit():
                    signals.sma_periods.add(int(period))
            except ValueError:
                continue
        
        # Detect crossovers
        signals.has_cross_up = bool(XOVER_RE.search(pine_code))
        signals.has_cross_down = bool(XUNDER_RE.search(pine_code))
        
        # Extract MACD parameters
        macd_match = MACD_RE.search(pine_code)
        if macd_match:
            try:
                _, fast, slow, signal = macd_match.groups()
                signals.macd = {
                    "fast": int(fast) if fast.isdigit() else 12,
                    "slow": int(slow) if slow.isdigit() else 26,
                    "signal": int(signal) if signal.isdigit() else 9
                }
            except ValueError:
                signals.macd = {"fast": 12, "slow": 26, "signal": 9}
        
        # Extract RSI periods
        for match in RSI_RE.finditer(pine_code):
            _, period = match.groups()
            try:
                if period.isdigit():
                    signals.rsi_periods.add(int(period))
            except ValueError:
                continue
        
        # Extract Bollinger Bands
        bb_match = BB_RE.search(pine_code)
        if bb_match:
            try:
                _, length, mult = bb_match.groups()[:3]
                signals.bollinger_bands = {
                    "length": int(length) if length.isdigit() else 20,
                    "multiplier": float(mult) if mult.replace('.', '').isdigit() else 2.0
                }
            except (ValueError, AttributeError):
                signals.bollinger_bands = {"length": 20, "multiplier": 2.0}
        
        # Extract higher timeframes
        for match in HTF_RE.finditer(pine_code):
            symbol, timeframe = match.groups()
            signals.higher_tf.append({
                "symbol": symbol.strip('\'"'),
                "tf": timeframe.strip('\'"')
            })
        
        # Check for FVG mentions
        for pattern in FVG_PATTERNS:
            if pattern.search(pine_code):
                signals.mentions_fvg = True
                break
        
        # Check for SMC mentions
        for pattern in SMC_PATTERNS:
            if pattern.search(pine_code):
                signals.mentions_smc = True
                break
        
        # Check for volume mentions
        for pattern in VOLUME_PATTERNS:
            if pattern.search(pine_code):
                signals.mentions_volume = True
                break
        
        # Check for support/resistance mentions
        for pattern in SR_PATTERNS:
            if pattern.search(pine_code):
                signals.mentions_sr = True
                break
        
        # Calculate complexity score
        signals.complexity_score = _calculate_complexity_score(signals, pine_code)
        
        # Determine strategy type
        signals.strategy_type = _determine_strategy_type(signals, pine_code)
        
        # Convert to dictionary
        result = {
            "ema_periods": sorted(list(signals.ema_periods)),
            "sma_periods": sorted(list(signals.sma_periods)),
            "has_cross_up": signals.has_cross_up,
            "has_cross_down": signals.has_cross_down,
            "macd": signals.macd,
            "rsi_periods": sorted(list(signals.rsi_periods)),
            "bollinger_bands": signals.bollinger_bands,
            "higher_tf": signals.higher_tf,
            "mentions_fvg": signals.mentions_fvg,
            "mentions_smc": signals.mentions_smc,
            "mentions_volume": signals.mentions_volume,
            "mentions_sr": signals.mentions_sr,
            "complexity_score": signals.complexity_score,
            "strategy_type": signals.strategy_type
        }
        
        logger.info(f"Extracted signals: {len(signals.ema_periods)} EMAs, "
                   f"{'MACD' if signals.macd else 'No MACD'}, "
                   f"{'SMC' if signals.mentions_smc else 'No SMC'}, "
                   f"Type: {signals.strategy_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract signals: {e}")
        return {
            "ema_periods": [],
            "sma_periods": [],
            "has_cross_up": False,
            "has_cross_down": False,
            "macd": None,
            "rsi_periods": [],
            "bollinger_bands": None,
            "higher_tf": [],
            "mentions_fvg": False,
            "mentions_smc": False,
            "mentions_volume": False,
            "mentions_sr": False,
            "complexity_score": 0.0,
            "strategy_type": "unknown"
        }

def _calculate_complexity_score(signals: ExtractedSignals, pine_code: str) -> float:
    """Calculate complexity score based on indicators and patterns"""
    score = 0.0
    
    # Base indicators
    score += len(signals.ema_periods) * 0.1
    score += len(signals.sma_periods) * 0.1
    score += len(signals.rsi_periods) * 0.15
    
    # Advanced indicators
    if signals.macd:
        score += 0.2
    if signals.bollinger_bands:
        score += 0.2
    
    # Pattern complexity
    if signals.mentions_smc:
        score += 0.3
    if signals.mentions_fvg:
        score += 0.25
    if signals.mentions_volume:
        score += 0.15
    if signals.mentions_sr:
        score += 0.1
    
    # Crossover complexity
    if signals.has_cross_up or signals.has_cross_down:
        score += 0.1
    
    # Higher timeframe analysis
    score += len(signals.higher_tf) * 0.15
    
    # Code length factor (rough complexity measure)
    code_lines = len(pine_code.split('\n'))
    if code_lines > 100:
        score += 0.2
    elif code_lines > 50:
        score += 0.1
    
    return round(score, 2)

def _determine_strategy_type(signals: ExtractedSignals, pine_code: str) -> str:
    """Determine strategy type based on patterns and indicators"""
    
    # SMC strategies
    if signals.mentions_smc or signals.mentions_fvg:
        return "smc"
    
    # Scalping indicators (short EMAs, frequent signals)
    short_emas = [p for p in signals.ema_periods if p <= 10]
    if short_emas and (signals.has_cross_up or signals.has_cross_down):
        return "scalping"
    
    # Swing trading indicators (longer EMAs)
    long_emas = [p for p in signals.ema_periods if p >= 50]
    if long_emas and not short_emas:
        return "swing"
    
    # Trend following
    if signals.ema_periods and (signals.has_cross_up or signals.has_cross_down):
        return "trending"
    
    # Mean reversion (BB, RSI)
    if signals.bollinger_bands or signals.rsi_periods:
        return "mean_reversion"
    
    # Momentum (MACD heavy)
    if signals.macd and not signals.ema_periods:
        return "momentum"
    
    # Range trading
    if signals.mentions_sr and not (signals.has_cross_up or signals.has_cross_down):
        return "ranging"
    
    # Multi-timeframe
    if len(signals.higher_tf) > 0:
        return "multi_timeframe"
    
    # Default
    return "trending" if signals.ema_periods else "indicator"

def extract_strategy_rules(pine_code: str) -> List[Dict[str, Any]]:
    """
    Extract trading rules and conditions from Pine Script
    
    Args:
        pine_code: Pine Script source code
        
    Returns:
        List of trading rules
    """
    rules = []
    
    try:
        # Look for strategy.entry calls
        entry_pattern = re.compile(r'strategy\.entry\(\s*["\']([^"\']+)["\']\s*,\s*strategy\.(long|short)', re.I)
        for match in entry_pattern.finditer(pine_code):
            rule_name, direction = match.groups()
            rules.append({
                "type": f"entry_{direction}",
                "name": rule_name,
                "action": "entry"
            })
        
        # Look for strategy.close calls
        close_pattern = re.compile(r'strategy\.close\(\s*["\']([^"\']+)["\']', re.I)
        for match in close_pattern.finditer(pine_code):
            rule_name = match.group(1)
            rules.append({
                "type": "exit",
                "name": rule_name,
                "action": "close"
            })
        
        # Look for conditions (if statements)
        condition_pattern = re.compile(r'if\s+([^{]+)\s*(?:{|\n)', re.I)
        for match in condition_pattern.finditer(pine_code):
            condition = match.group(1).strip()
            if any(word in condition.lower() for word in ['crossover', 'crossunder', '>', '<', 'and', 'or']):
                rules.append({
                    "type": "condition",
                    "condition": condition,
                    "action": "evaluate"
                })
        
        logger.info(f"Extracted {len(rules)} trading rules")
        return rules
        
    except Exception as e:
        logger.error(f"Failed to extract rules: {e}")
        return []

def normalize_pine_code(pine_code: str) -> str:
    """
    Normalize Pine Script code for analysis
    
    Args:
        pine_code: Raw Pine Script code
        
    Returns:
        Normalized code
    """
    try:
        lines = pine_code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove comments
            line = re.sub(r'//.*$', '', line)
            
            # Remove extra whitespace
            line = line.strip()
            
            # Skip empty lines
            if line:
                normalized_lines.append(line)
        
        normalized = '\n'.join(normalized_lines)
        logger.info(f"Normalized Pine Script: {len(normalized_lines)} lines")
        return normalized
        
    except Exception as e:
        logger.error(f"Failed to normalize Pine code: {e}")
        return pine_code

def analyze_pine_script(pine_code: str) -> Dict[str, Any]:
    """
    Complete Pine Script analysis
    
    Args:
        pine_code: Pine Script source code
        
    Returns:
        Complete analysis results
    """
    try:
        normalized_code = normalize_pine_code(pine_code)
        inputs = extract_inputs(normalized_code)
        signals = extract_signals(normalized_code)
        rules = extract_strategy_rules(normalized_code)
        
        analysis = {
            "inputs": inputs,
            "signals": signals,
            "rules": rules,
            "code_stats": {
                "original_lines": len(pine_code.split('\n')),
                "normalized_lines": len(normalized_code.split('\n')),
                "characters": len(pine_code),
                "complexity_score": signals.get("complexity_score", 0.0)
            },
            "analysis_complete": True
        }
        
        logger.info(f"Complete Pine Script analysis: {len(inputs)} inputs, "
                   f"{len(signals.get('ema_periods', []))} EMAs, "
                   f"{len(rules)} rules, "
                   f"Type: {signals.get('strategy_type', 'unknown')}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")
        return {
            "inputs": [],
            "signals": {},
            "rules": [],
            "code_stats": {"error": str(e)},
            "analysis_complete": False
        }