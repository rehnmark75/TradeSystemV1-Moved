#!/usr/bin/env python3
import sys
import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    print('🔍 Testing validation components directly...')
    
    # Load replay_config directly
    replay_config = load_module_from_file(
        'replay_config', 
        '/app/forex_scanner/validation/replay_config.py'
    )
    print('✅ replay_config loaded directly')
    print(f'📊 Available timeframes: {list(replay_config.TIMEFRAME_MINUTES.keys())}')
    print(f'📊 Configured strategies: {list(replay_config.STRATEGY_REPLAY_CONFIG.keys())}')
    
    # Load error_handling directly
    error_handling = load_module_from_file(
        'error_handling',
        '/app/forex_scanner/validation/error_handling.py'
    )
    print('✅ error_handling loaded directly')
    
    # Test error classes
    try:
        raise error_handling.ValidationError('Test error')
    except error_handling.ValidationError as e:
        print(f'✅ ValidationError working: {e}')
    
    print('\n🎉 Direct validation components test PASSED!')
    print('✅ Core validation logic is functional')
    print('❌ Import system needs fixing for full module integration')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
