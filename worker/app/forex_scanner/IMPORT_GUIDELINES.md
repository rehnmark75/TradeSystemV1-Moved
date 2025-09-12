# Import Pattern Guidelines for Forex Scanner

## Overview
This document establishes standardized import patterns for the forex scanner to prevent cascade failures when modules are executed in different contexts (e.g., validation system, debugging scripts, direct execution).

## The Problem
Import path assumptions can break when:
- Running modules from different directories
- Validation system changes execution context
- Direct script execution vs module execution
- Docker vs local execution contexts

## Standardized Fallback Pattern

### For System Config Imports
```python
try:
    import config
except ImportError:
    from forex_scanner import config
```

### For ConfigData Imports
```python
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config
```

### For Relative Module Imports
```python
try:
    from .module_name import ClassName
except ImportError:
    from forex_scanner.package.module_name import ClassName
```

### For Core Module Imports
```python
try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.signal_detector import SignalDetector
```

### For Utils and Analysis Imports
```python
try:
    from utils.scanner_utils import make_json_serializable
    from analysis.technical import TechnicalAnalyzer
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable
    from forex_scanner.analysis.technical import TechnicalAnalyzer
```

## Pattern Implementation Rules

### 1. Always Use Try/Except Blocks
- Never use hardcoded absolute or relative imports alone
- Always provide both primary and fallback import paths
- Handle ImportError specifically, not generic exceptions

### 2. Order of Precedence
1. **Primary**: Relative imports (for normal module execution)
2. **Fallback**: Absolute imports with forex_scanner prefix

### 3. Group Related Imports
```python
# Good: Group related imports in same try/except block
try:
    from core.database import DatabaseManager
    from core.signal_detector import SignalDetector
    from core.data_fetcher import DataFetcher
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner.core.data_fetcher import DataFetcher
```

### 4. Component Availability Pattern
For optional components that may not exist:
```python
try:
    from core.trading.trade_validator import TradeValidator
    TRADE_VALIDATOR_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.trading.trade_validator import TradeValidator
        TRADE_VALIDATOR_AVAILABLE = True
    except ImportError:
        TRADE_VALIDATOR_AVAILABLE = False
        TradeValidator = None
        logging.getLogger(__name__).warning("TradeValidator not available")
```

## File-Specific Patterns

### Strategy Files
```python
# core/strategies/example_strategy.py
try:
    from configdata import config
    from .base_strategy import BaseStrategy
    from ..detection.price_adjuster import PriceAdjuster
except ImportError:
    from forex_scanner.configdata import config
    from forex_scanner.core.strategies.base_strategy import BaseStrategy
    from forex_scanner.core.detection.price_adjuster import PriceAdjuster
```

### Helper Files
```python
# core/strategies/helpers/example_helper.py
try:
    import config
    from configdata import config as configdata
except ImportError:
    from forex_scanner import config
    from forex_scanner.configdata import config as configdata
```

### Core System Files
```python
# core/scanner.py
try:
    import config
    from core.signal_detector import SignalDetector
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner.core.database import DatabaseManager
```

## Validation and Testing

### Import Validation Script
```python
def test_imports():
    """Test all critical imports work in both contexts"""
    try:
        # Test core modules
        from forex_scanner.core.scanner import ForexScanner
        from forex_scanner.core.trading.trading_orchestrator import TradingOrchestrator
        from forex_scanner.alerts.claude_analyzer import ClaudeAnalyzer
        print("✅ All critical imports working")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
```

### Component Availability Test
```python
def test_component_availability():
    """Test that components gracefully handle missing dependencies"""
    from forex_scanner.core.trading.trading_orchestrator import TradingOrchestrator
    
    orchestrator = TradingOrchestrator()
    components = orchestrator.check_component_availability()
    
    # Should not raise exceptions even if components missing
    assert isinstance(components, dict)
    print(f"✅ Component availability check passed: {components}")
```

## Common Pitfalls to Avoid

### 1. Nested Try/Except Anti-Pattern
```python
# Bad: Overly nested
try:
    try:
        from .module import Class
    except ImportError:
        from forex_scanner.module import Class
except ImportError:
    Class = None

# Good: Clean single-level fallback
try:
    from .module import Class
except ImportError:
    from forex_scanner.module import Class
```

### 2. Missing Fallback Paths
```python
# Bad: No fallback
from core.database import DatabaseManager

# Good: With fallback
try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager
```

### 3. Inconsistent Import Styles
```python
# Bad: Mixed styles in same file
import config  # hardcoded
from forex_scanner.core import database  # hardcoded absolute

# Good: Consistent fallback pattern
try:
    import config
    from core import database
except ImportError:
    from forex_scanner import config
    from forex_scanner.core import database
```

## Enforcement and Monitoring

### 1. Automated Checking
- Grep patterns to find hardcoded imports: `grep -r "^import config$\|^from core\." --include="*.py"`
- CI checks to validate import patterns
- Regular audits of new files

### 2. Development Guidelines
- All new files MUST use standardized fallback patterns
- Code review should verify proper import patterns
- Update this document when new patterns are needed

### 3. Migration Strategy
- Use automated scripts to convert existing files
- Prioritize critical system files first
- Test imports after each conversion batch

## Benefits Achieved

1. **Cascade Failure Prevention**: No more system-wide breakage from context changes
2. **Execution Context Independence**: Works in validation, direct execution, Docker, etc.
3. **Graceful Degradation**: Components handle missing dependencies cleanly
4. **Maintainability**: Consistent patterns across entire codebase
5. **Developer Confidence**: Predictable import behavior reduces debugging time

## Success Metrics

A properly hardened system should show:
- ✅ All core modules importable from any context
- ✅ No ImportError exceptions during normal operation
- ✅ Graceful component unavailability warnings instead of crashes
- ✅ Validation system works without breaking live scanner
- ✅ Direct script execution works from any directory

---

**Last Updated**: 2025-09-04  
**Status**: Active - All critical files converted to standardized patterns  
**Coverage**: 95%+ of core system files hardened