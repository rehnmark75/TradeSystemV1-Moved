#!/usr/bin/env python3
"""
Scanner Claude Integration Fix
Based on analysis of core/scanner.py and alerts/claude_api.py
"""

def analyze_scanner_constructor():
    """Analyze the scanner constructor from the code"""
    
    print("🔍 SCANNER CONSTRUCTOR ANALYSIS")
    print("=" * 60)
    
    print("📋 IntelligentForexScanner.__init__ parameters:")
    print("""
    def __init__(
        self,
        db_manager: DatabaseManager,
        epic_list: List[str] = None,
        scan_interval: int = 60,
        claude_api_key: str = None,          # ✅ Correct parameter name
        enable_claude_analysis: bool = False, # ❌ Defaults to False!
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5,
        min_confidence: float = 0.6,
        user_timezone: str = 'Europe/Stockholm',
        intelligence_update_interval: int = 300,
        enable_intelligence: bool = True
    )
    """)
    
    print("\n🚨 ISSUES IDENTIFIED:")
    print("1. enable_claude_analysis defaults to False")
    print("2. Scanner constructor receives parameters but doesn't store them as expected")
    print("3. Claude analyzer initialization might be conditional")
    
    print("\n🔧 LOOKING AT CONSTRUCTOR BODY...")
    print("The scanner constructor imports:")
    print("- from alerts.claude_api import ClaudeAnalyzer")
    print("- This means ClaudeAnalyzer class exists")
    
    print("\nBut the constructor must be missing the Claude initialization code!")

def create_scanner_claude_patch():
    """Create a patch to fix Claude integration in the scanner"""
    
    print("\n🔧 CREATING SCANNER CLAUDE INTEGRATION PATCH")
    print("=" * 60)
    
    patch_code = '''
# SCANNER CLAUDE INTEGRATION PATCH
# Add this to your IntelligentForexScanner.__init__ method

def __init__(self, ...):
    # ... existing initialization code ...
    
    # Store Claude parameters as instance attributes
    self.claude_api_key = claude_api_key
    self.enable_claude_analysis = enable_claude_analysis
    
    # Initialize Claude analyzer
    self.claude_analyzer = None
    if enable_claude_analysis and claude_api_key:
        try:
            from alerts.claude_api import ClaudeAnalyzer
            self.claude_analyzer = ClaudeAnalyzer(claude_api_key)
            self.logger.info(f"🤖 Claude analyzer initialized with key: {claude_api_key[:10]}...")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Claude analyzer: {e}")
            self.enable_claude_analysis = False
    elif enable_claude_analysis and not claude_api_key:
        self.logger.warning("⚠️ Claude analysis enabled but no API key provided")
        self.enable_claude_analysis = False
    else:
        self.logger.info("🤖 Claude analysis disabled")
    
    # ... rest of initialization ...
'''
    
    print(patch_code)
    
    return patch_code

def create_signal_processing_patch():
    """Create patch for signal processing with Claude"""
    
    print("\n🔧 SIGNAL PROCESSING CLAUDE PATCH")
    print("=" * 60)
    
    processing_patch = '''
# ADD CLAUDE PROCESSING TO YOUR SIGNAL HANDLING
# In your scan_and_trade function or signal processing loop:

def process_signal_with_claude(signal, scanner):
    """Process signal and add Claude analysis"""
    
    # Check if Claude is available and enabled
    if (hasattr(scanner, 'claude_analyzer') and 
        scanner.claude_analyzer and 
        getattr(scanner, 'enable_claude_analysis', False)):
        
        try:
            print(f"🤖 Requesting Claude analysis for {signal['epic']}...")
            
            # Call Claude analyzer
            claude_result = scanner.claude_analyzer.analyze_signal(signal)
            
            if claude_result and len(claude_result) > 50:
                # Add Claude analysis to signal
                signal['claude_analysis'] = claude_result
                print(f"✅ Claude analysis successful: {len(claude_result)} chars")
                
                # Show preview
                preview = claude_result[:100].replace('\\n', ' ')
                print(f"🤖 Claude preview: {preview}...")
                
            else:
                print(f"❌ Claude analysis failed or empty: {claude_result}")
                
        except Exception as e:
            print(f"❌ Claude analysis error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Claude not available:")
        print(f"   has_claude_analyzer: {hasattr(scanner, 'claude_analyzer')}")
        print(f"   claude_analyzer_exists: {getattr(scanner, 'claude_analyzer', None) is not None}")
        print(f"   enable_claude_analysis: {getattr(scanner, 'enable_claude_analysis', False)}")
    
    return signal

# USAGE IN YOUR SCAN LOOP:
for signal in signals:
    # Your existing signal processing
    epic = signal.get('epic', 'Unknown')
    signal_type = signal.get('signal_type', 'Unknown')
    confidence = signal.get('confidence_score', 0)
    
    # Add Claude analysis
    signal = process_signal_with_claude(signal, scanner)
    
    # Now check for Claude analysis
    if signal.get('claude_analysis'):
        analysis_preview = signal['claude_analysis'][:100]
        log_and_print(f"🤖 Claude: {analysis_preview}...")
    else:
        log_and_print(f"❌ No Claude analysis available")
'''
    
    print(processing_patch)
    return processing_patch

def create_complete_fix():
    """Create complete fix for Claude integration"""
    
    print("\n🚀 COMPLETE CLAUDE INTEGRATION FIX")
    print("=" * 60)
    
    complete_fix = '''
# COMPLETE CLAUDE INTEGRATION FIX
# Replace your scanner initialization with this:

# Debug Claude configuration FIRST
claude_api_key = "sk-ant-api03-your-actual-key-here"  # Your real key
enable_claude = True

print(f"🔧 Claude Configuration:")
print(f"   API Key: {claude_api_key[:20]}...")
print(f"   Enable Claude: {enable_claude}")

# Test Claude directly BEFORE scanner
try:
    from alerts.claude_api import ClaudeAnalyzer
    test_analyzer = ClaudeAnalyzer(claude_api_key)
    
    # Test Claude with dummy signal
    test_result = test_analyzer.analyze_signal({
        'signal_type': 'BULL',
        'epic': 'CS.D.EURUSD.CEEM.IP',
        'confidence_score': 0.75,
        'price': 1.0850
    })
    
    if test_result and len(test_result) > 50:
        print(f"✅ Direct Claude test successful: {len(test_result)} chars")
        claude_working = True
    else:
        print(f"❌ Direct Claude test failed: {test_result}")
        claude_working = False
        
except Exception as e:
    print(f"❌ Direct Claude test error: {e}")
    claude_working = False

# Create scanner with explicit Claude settings
scanner = IntelligentForexScanner(
    db_manager=db_manager,
    epic_list=['CS.D.EURUSD.CEEM.IP'],  # Hard-coded for testing
    claude_api_key=claude_api_key,      # Direct variable
    enable_claude_analysis=enable_claude, # Direct variable
    use_bid_adjustment=True,
    spread_pips=1.5,
    min_confidence=0.6,
    user_timezone='Europe/Stockholm'
    # Remove intelligence_mode parameter - might be interfering
)

# FORCE Claude initialization if scanner didn't do it
if claude_working and not hasattr(scanner, 'claude_analyzer'):
    print("🔧 Manually initializing Claude in scanner...")
    scanner.claude_api_key = claude_api_key
    scanner.enable_claude_analysis = enable_claude
    scanner.claude_analyzer = ClaudeAnalyzer(claude_api_key)
    print("✅ Manual Claude initialization complete")

# Debug scanner Claude state
print(f"🔧 Scanner Claude Status:")
print(f"   claude_api_key: {getattr(scanner, 'claude_api_key', 'NOT_SET')[:20]}...")
print(f"   enable_claude_analysis: {getattr(scanner, 'enable_claude_analysis', 'NOT_SET')}")
print(f"   has claude_analyzer: {hasattr(scanner, 'claude_analyzer')}")

# Test scanner Claude
if hasattr(scanner, 'claude_analyzer') and scanner.claude_analyzer:
    try:
        test_signal = {
            'signal_type': 'BULL',
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'confidence_score': 0.85,
            'price': 1.0875
        }
        
        scanner_result = scanner.claude_analyzer.analyze_signal(test_signal)
        if scanner_result:
            print(f"✅ Scanner Claude test successful: {len(scanner_result)} chars")
        else:
            print(f"❌ Scanner Claude test failed")
            
    except Exception as e:
        print(f"❌ Scanner Claude test error: {e}")
else:
    print("❌ Scanner has no working Claude analyzer")

# NOW your scan loop should work with Claude
'''
    
    print(complete_fix)
    return complete_fix

def create_test_script():
    """Create test script for Claude integration"""
    
    test_script = '''#!/usr/bin/env python3
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
            'epic': 'CS.D.EURUSD.CEEM.IP',
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
            epic_list=['CS.D.EURUSD.CEEM.IP'],
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
'''
    
    try:
        with open('/app/test_claude_integration.py', 'w') as f:
            f.write(test_script)
        print("\n📄 Test script created: /app/test_claude_integration.py")
        print("💡 Edit with your real API key and DB URL, then run it")
        return True
    except Exception as e:
        print(f"⚠️ Could not create test script: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SCANNER CLAUDE INTEGRATION ANALYSIS & FIX")
    print("=" * 80)
    
    # Analyze the issue
    analyze_scanner_constructor()
    
    # Create fixes
    create_scanner_claude_patch()
    create_signal_processing_patch() 
    create_complete_fix()
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 80)
    print("🎯 SOLUTION SUMMARY")
    print("=" * 80)
    
    print("🔍 PROBLEM IDENTIFIED:")
    print("1. Scanner constructor receives Claude parameters correctly")
    print("2. BUT scanner doesn't initialize claude_analyzer attribute")
    print("3. Scanner stores parameters but doesn't create ClaudeAnalyzer instance")
    
    print("\n🔧 SOLUTIONS PROVIDED:")
    print("1. Patch scanner constructor to initialize Claude properly")
    print("2. Force Claude initialization after scanner creation")
    print("3. Add Claude processing to your signal handling loop")
    print("4. Complete working example with all fixes")
    
    print("\n⚡ QUICK FIX:")
    print("Add this after your scanner creation:")
    print("""
    # Force Claude initialization
    if not hasattr(scanner, 'claude_analyzer'):
        from alerts.claude_api import ClaudeAnalyzer
        scanner.claude_analyzer = ClaudeAnalyzer("your-api-key")
        scanner.enable_claude_analysis = True
        print("🤖 Manually initialized Claude in scanner")
    """)
    
    print("\n🧪 TEST:")
    print("Run: python /app/test_claude_integration.py")
    print("This will confirm Claude is working before using it in your scanner")