#!/usr/bin/env python3
"""
Claude File Saving Diagnostic Script
Run this to debug why Claude analysis files aren't being saved
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def test_claude_file_saving():
    """Test Claude file saving step by step"""
    print("🔍 Debugging Claude File Saving...")
    print("=" * 50)
    
    # Step 1: Check directory and permissions
    print("Step 1: Checking directory and permissions")
    
    current_dir = os.getcwd()
    claude_dir = os.path.join(current_dir, "claude_analysis")
    
    print(f"   Current directory: {current_dir}")
    print(f"   Claude directory: {claude_dir}")
    print(f"   Directory exists: {os.path.exists(claude_dir)}")
    
    # Test directory creation
    try:
        os.makedirs(claude_dir, exist_ok=True)
        print("   ✅ Directory creation: SUCCESS")
    except Exception as e:
        print(f"   ❌ Directory creation: FAILED - {e}")
        return False
    
    # Test file writing
    test_file = os.path.join(claude_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write(f"Test file created at {datetime.now()}")
        print("   ✅ File writing: SUCCESS")
        os.remove(test_file)
        print("   ✅ File cleanup: SUCCESS")
    except Exception as e:
        print(f"   ❌ File writing: FAILED - {e}")
        return False
    
    # Step 2: Check imports
    print("\nStep 2: Checking imports")
    
    try:
        import config
        print("   ✅ Config import: SUCCESS")
        api_key = getattr(config, 'CLAUDE_API_KEY', None)
        print(f"   Claude API key present: {'YES' if api_key else 'NO'}")
    except Exception as e:
        print(f"   ❌ Config import: FAILED - {e}")
        return False
    
    try:
        from alerts import ClaudeAnalyzer
        print("   ✅ ClaudeAnalyzer import: SUCCESS")
    except Exception as e:
        print(f"   ❌ ClaudeAnalyzer import: FAILED - {e}")
        return False
    
    # Step 3: Test ClaudeAnalyzer initialization
    print("\nStep 3: Testing ClaudeAnalyzer initialization")
    
    try:
        analyzer = ClaudeAnalyzer(api_key=api_key, auto_save=True, save_directory="claude_analysis")
        print("   ✅ ClaudeAnalyzer creation: SUCCESS")
        print(f"   Implementation: {getattr(analyzer, 'implementation', 'Unknown')}")
        
        # Check if it has the analyzer attribute
        if hasattr(analyzer, 'analyzer'):
            print(f"   Analyzer attribute: {type(analyzer.analyzer).__name__}")
            
            # Check file_manager
            if hasattr(analyzer.analyzer, 'file_manager'):
                file_manager = analyzer.analyzer.file_manager
                print(f"   FileManager present: YES")
                print(f"   FileManager auto_save: {getattr(file_manager, 'auto_save', 'Unknown')}")
                print(f"   FileManager save_directory: {getattr(file_manager, 'save_directory', 'Unknown')}")
            else:
                print("   FileManager present: NO")
        else:
            print("   Analyzer attribute: NO")
            
    except Exception as e:
        print(f"   ❌ ClaudeAnalyzer creation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test manual file saving
    print("\nStep 4: Testing manual file saving")
    
    test_signal = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BUY',
        'confidence_score': 0.85,
        'price': 1.1234,
        'strategy': 'EMA',
        'timestamp': datetime.now().isoformat()
    }
    
    test_analysis = {
        'score': 8,
        'decision': 'APPROVE',
        'approved': True,
        'reason': 'Test analysis',
        'raw_response': 'SCORE: 8\nDECISION: APPROVE\nREASON: Test analysis',
        'technical_validation_passed': True
    }
    
    try:
        # Try to access FileManager directly
        if hasattr(analyzer, 'analyzer') and hasattr(analyzer.analyzer, 'file_manager'):
            file_manager = analyzer.analyzer.file_manager
            
            print("   Testing FileManager.save_minimal_analysis...")
            file_manager.save_minimal_analysis(test_signal, test_analysis)
            
            # Check if file was created
            files = os.listdir(claude_dir)
            claude_files = [f for f in files if f.startswith('minimal_analysis_')]
            
            if claude_files:
                print(f"   ✅ File created: {claude_files[-1]}")
                
                # Show file content
                with open(os.path.join(claude_dir, claude_files[-1]), 'r') as f:
                    content = f.read()
                    print(f"   File size: {len(content)} characters")
                    print(f"   First 200 chars: {content[:200]}...")
                return True
            else:
                print("   ❌ No file created")
                return False
        else:
            print("   ❌ FileManager not accessible")
            return False
            
    except Exception as e:
        print(f"   ❌ Manual file saving: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyze_signal_minimal():
    """Test the full analyze_signal_minimal flow"""
    print("\n" + "=" * 50)
    print("🧪 Testing Full analyze_signal_minimal Flow")
    print("=" * 50)
    
    try:
        import config
        from alerts import ClaudeAnalyzer
        
        api_key = getattr(config, 'CLAUDE_API_KEY', None)
        if not api_key:
            print("❌ No Claude API key - cannot test full flow")
            return False
        
        analyzer = ClaudeAnalyzer(api_key=api_key, auto_save=True, save_directory="claude_analysis")
        
        test_signal = {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'signal_type': 'BUY',
            'confidence_score': 0.85,
            'price': 1.1234,
            'strategy': 'EMA',
            'timestamp': datetime.now().isoformat()
        }
        
        print("Calling analyze_signal_minimal with save_to_file=True...")
        
        # Enable debug logging temporarily
        logging.basicConfig(level=logging.DEBUG)
        
        result = analyzer.analyze_signal_minimal(test_signal, save_to_file=True)
        
        if result:
            print(f"✅ Analysis result: {result.get('decision')} ({result.get('score')}/10)")
            
            # Check for files
            claude_dir = "claude_analysis"
            if os.path.exists(claude_dir):
                files = os.listdir(claude_dir)
                recent_files = [f for f in files if f.startswith('minimal_analysis_')]
                
                if recent_files:
                    print(f"✅ Files found: {len(recent_files)}")
                    for f in recent_files[-3:]:  # Show last 3 files
                        print(f"   - {f}")
                    return True
                else:
                    print("❌ No analysis files found despite successful analysis")
                    return False
            else:
                print("❌ Claude analysis directory doesn't exist")
                return False
        else:
            print("❌ Analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Full flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 Claude File Saving Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Basic file operations
    basic_test = test_claude_file_saving()
    
    # Test 2: Full analysis flow (if API key available)
    if basic_test:
        full_test = test_analyze_signal_minimal()
    else:
        print("\n❌ Skipping full test due to basic test failure")
        full_test = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Basic file operations: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Full analysis flow: {'✅ PASS' if full_test else '❌ FAIL'}")
    
    if basic_test and full_test:
        print("\n🎉 All tests passed! File saving should work.")
    elif basic_test and not full_test:
        print("\n⚠️ Basic operations work, but full flow fails. Check API or implementation.")
    else:
        print("\n❌ Basic operations failed. Check permissions and setup.")
    
    # Show current directory contents
    print(f"\nCurrent directory: {os.getcwd()}")
    if os.path.exists("claude_analysis"):
        files = os.listdir("claude_analysis")
        print(f"Claude analysis files: {len(files)}")
        for f in files:
            print(f"  - {f}")
    else:
        print("Claude analysis directory: NOT FOUND")

if __name__ == "__main__":
    main()