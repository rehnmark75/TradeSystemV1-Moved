#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
Run this to check if the module structure is set up properly
"""

def test_imports():
    """Test all module imports"""
    
    print("🧪 Testing Forex Scanner imports...")
    
    try:
        print("  ✓ Testing config import...")
        import config
        print(f"    - Epic list: {len(config.EPIC_LIST)} pairs")
        
        print("  ✓ Testing core modules...")
        from core.database import DatabaseManager
        from core.data_fetcher import DataFetcher
        from core.signal_detector import SignalDetector
        from core.scanner import ForexScanner
        
        print("  ✓ Testing analysis modules...")
        from analysis.technical import TechnicalAnalyzer
        from analysis.volume import VolumeAnalyzer
        from analysis.behavior import BehaviorAnalyzer
        from analysis.multi_timeframe import MultiTimeframeAnalyzer
        
        print("  ✓ Testing alert modules...")
        from alerts.claude_api import ClaudeAnalyzer
        from alerts.notifications import NotificationManager
        
        print("  ✓ Testing utility modules...")
        from utils.helpers import setup_logging, extract_pair_from_epic
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without database"""
    
    print("\n🔧 Testing basic functionality...")
    
    try:
        from utils.helpers import extract_pair_from_epic, get_pip_multiplier
        
        # Test utility functions
        pair = extract_pair_from_epic('CS.D.EURUSD.MINI.IP')
        print(f"  ✓ Pair extraction: {pair}")
        
        pip_mult = get_pip_multiplier('EURUSD')
        print(f"  ✓ Pip multiplier: {pip_mult}")
        
        # Test analyzers (without data)
        from analysis.technical import TechnicalAnalyzer
        from analysis.volume import VolumeAnalyzer
        
        tech_analyzer = TechnicalAnalyzer()
        vol_analyzer = VolumeAnalyzer()
        print("  ✓ Analyzers initialized")
        
        # Test Claude analyzer (without API key)
        from alerts.claude_api import ClaudeAnalyzer
        claude = ClaudeAnalyzer("")  # Empty API key
        print("  ✓ Claude analyzer initialized (no API key)")
        
        print("🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Forex Scanner Structure Test")
    print("=" * 50)
    
    import_success = test_imports()
    
    if import_success:
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n✅ All tests passed! Structure is ready.")
            print("\nNext steps:")
            print("1. Configure your database URL in config.py")
            print("2. Add your Claude API key to config.py")
            print("3. Run: python main.py scan --config-check")
        else:
            print("\n⚠️ Import tests passed but functionality tests failed")
    else:
        print("\n❌ Import tests failed. Check your directory structure:")
        print("- Ensure all __init__.py files exist")
        print("- Verify file locations match the expected structure")
        print("- Check for typos in file names")


if __name__ == '__main__':
    main()