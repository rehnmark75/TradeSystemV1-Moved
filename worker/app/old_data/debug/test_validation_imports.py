#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

print('🔍 Testing validation system imports after core fixes...')

try:
    # Test the problematic import chain
    print('1. Testing core.scanner import...')
    from forex_scanner.core.scanner import IntelligentForexScanner
    print('✅ core.scanner imported successfully')
    
    print('2. Testing core.database import...')
    from forex_scanner.core.database import DatabaseManager
    print('✅ core.database imported successfully')
    
    print('3. Testing validation.historical_data_manager import...')
    from forex_scanner.validation.historical_data_manager import HistoricalDataManager
    print('✅ validation.historical_data_manager imported successfully')
    
    print('4. Testing full validation system import...')
    from forex_scanner.validation.signal_replay_validator import SignalReplayValidator
    print('✅ Full validation system imported successfully')
    
    print('\n🎉 ALL VALIDATION IMPORTS SUCCESSFUL!')
    print('✅ The core scanner import fixes resolved the validation system issues')
    
except ImportError as e:
    print(f'❌ Import still failing: {e}')
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f'❌ Other error: {e}')
    import traceback
    traceback.print_exc()
