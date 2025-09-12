# commands/__init__.py
"""
Command modules for Forex Scanner CLI
Modular command organization for better maintainability
"""

from .scanner_commands import ScannerCommands
from .backtest_commands import BacktestCommands  
from .debug_commands import DebugCommands
from .scalping_commands import ScalpingCommands
from .claude_commands import ClaudeCommands
from .analysis_commands import AnalysisCommands

__all__ = [
    'ScannerCommands',
    'BacktestCommands',
    'DebugCommands', 
    'ScalpingCommands',
    'ClaudeCommands',
    'AnalysisCommands'
]

# Version info
__version__ = "1.0.0"
__author__ = "Forex Scanner Team"

def get_available_command_modules():
    """Get list of available command modules"""
    return [
        'ScannerCommands',
        'BacktestCommands',
        'DebugCommands',
        'ScalpingCommands', 
        'ClaudeCommands',
        'AnalysisCommands'
    ]

def validate_command_imports():
    """Validate that all command modules import correctly"""
    import_status = {}
    
    try:
        from .scanner_commands import ScannerCommands
        import_status['ScannerCommands'] = True
    except ImportError as e:
        import_status['ScannerCommands'] = str(e)
    
    try:
        from .backtest_commands import BacktestCommands
        import_status['BacktestCommands'] = True
    except ImportError as e:
        import_status['BacktestCommands'] = str(e)
    
    try:
        from .debug_commands import DebugCommands
        import_status['DebugCommands'] = True
    except ImportError as e:
        import_status['DebugCommands'] = str(e)
    
    try:
        from .scalping_commands import ScalpingCommands
        import_status['ScalpingCommands'] = True
    except ImportError as e:
        import_status['ScalpingCommands'] = str(e)
    
    try:
        from .claude_commands import ClaudeCommands
        import_status['ClaudeCommands'] = True
    except ImportError as e:
        import_status['ClaudeCommands'] = str(e)
    
    try:
        from .analysis_commands import AnalysisCommands
        import_status['AnalysisCommands'] = True
    except ImportError as e:
        import_status['AnalysisCommands'] = str(e)
    
    return import_status

if __name__ == "__main__":
    print("🔍 Command Module Import Status:")
    print("=" * 50)
    
    status = validate_command_imports()
    
    for module, result in status.items():
        if result is True:
            print(f"✅ {module}")
        else:
            print(f"❌ {module}: {result}")
    
    successful = sum(1 for result in status.values() if result is True)
    total = len(status)
    
    print("=" * 50)
    print(f"📊 Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
