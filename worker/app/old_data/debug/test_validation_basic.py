#!/usr/bin/env python3
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/app')

try:
    print('🔍 Testing validation system imports...')
    
    # Test basic imports
    from forex_scanner.validation.replay_config import ReplayConfig
    print('✅ replay_config imported successfully')
    
    from forex_scanner.validation.error_handling import ValidationError
    print('✅ error_handling imported successfully')
    
    from forex_scanner.validation.validation_reporter import ValidationReporter, ValidationResult
    print('✅ validation_reporter imported successfully')
    
    # Test basic functionality
    config = ReplayConfig()
    print(f'✅ ReplayConfig initialized: {len(config.__dict__)} settings')
    
    reporter = ValidationReporter(use_colors=False)
    print('✅ ValidationReporter initialized')
    
    # Create a dummy validation result
    result = ValidationResult(
        success=True,
        epic='CS.D.EURUSD.MINI.IP',
        timestamp='2025-01-15T14:30:00Z',
        signal_detected=True,
        processing_time_ms=250.0,
        message='Test validation successful'
    )
    print('✅ ValidationResult created')
    
    # Test report generation
    report = reporter.generate_text_report([result])
    print('✅ Report generation successful')
    print(f'📊 Report length: {len(report)} characters')
    
    print('\n🎉 Basic validation system test PASSED!')
    print('✅ All core validation components are working')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('🔧 Some validation modules may need import path fixes')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    print('🔧 Validation system may need debugging')
    sys.exit(1)
