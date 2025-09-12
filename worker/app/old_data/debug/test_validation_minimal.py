#!/usr/bin/env python3
import sys

try:
    print('🔍 Testing minimal validation components...')
    
    # Test standalone modules first
    from forex_scanner.validation.replay_config import ReplayConfig, TIMEFRAME_MINUTES, STRATEGY_REPLAY_CONFIG
    print('✅ replay_config imported successfully')
    
    from forex_scanner.validation.error_handling import ValidationError, DataRetrievalError
    print('✅ error_handling imported successfully')
    
    # Test basic functionality
    print(f'📊 Available timeframes: {list(TIMEFRAME_MINUTES.keys())}')
    print(f'📊 Configured strategies: {list(STRATEGY_REPLAY_CONFIG.keys())}')
    
    # Test error classes
    try:
        raise ValidationError('Test error')
    except ValidationError as e:
        print(f'✅ ValidationError working: {e}')
    
    print('\n🎉 Minimal validation components test PASSED!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
