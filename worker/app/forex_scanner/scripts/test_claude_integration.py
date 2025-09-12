#!/usr/bin/env python3
"""
Test Claude Integration in Scanner
"""

import sys
sys.path.insert(0, '/app/forex_scanner')

def test_claude_integration():
    print("🧪 TESTING CLAUDE INTEGRATION")
    print("=" * 50)
    
    try:
        # Test imports
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        from alerts.claude_api import ClaudeAnalyzer
        print("✅ All imports successful")
        
        # Test Claude directly
        api_key = "sk-ant-api03-your-actual-key-here"  # Your real key
        analyzer = ClaudeAnalyzer(api_key)
        
        test_signal = {
            'signal_type': 'BULL',
            'epic': 'CS.D.EURUSD.MINI.IP',
            'confidence_score': 0.75,
            'price': 1.0850
        }
        
        result = analyzer.analyze_signal(test_signal)
        if result:
            print(f"✅ Direct Claude working: {len(result)} chars")
        else:
            print("❌ Direct Claude failed")
            return False
        
        # Test scanner with Claude
        db_manager = DatabaseManager("your-db-url")  # Your real DB URL
        
        scanner = IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=['CS.D.EURUSD.MINI.IP'],
            claude_api_key=api_key,
            enable_claude_analysis=True
        )
        
        # Force Claude if not initialized
        if not hasattr(scanner, 'claude_analyzer'):
            scanner.claude_analyzer = analyzer
            scanner.enable_claude_analysis = True
            print("🔧 Manually added Claude to scanner")
        
        # Test scanner Claude
        if hasattr(scanner, 'claude_analyzer'):
            scanner_result = scanner.claude_analyzer.analyze_signal(test_signal)
            if scanner_result:
                print(f"✅ Scanner Claude working: {len(scanner_result)} chars")
                return True
            else:
                print("❌ Scanner Claude failed")
                return False
        else:
            print("❌ Scanner has no Claude analyzer")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_claude_integration()
