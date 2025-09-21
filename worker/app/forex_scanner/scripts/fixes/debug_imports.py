#!/usr/bin/env python3
"""
Debug script to check logging and scanner activity
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add forex_scanner to path
sys.path.append('/app/forex_scanner')

# Create missing __init__.py if needed
if not os.path.exists('/app/forex_scanner/__init__.py'):
    open('/app/forex_scanner/__init__.py', 'w').close()

def test_logging():
    """Test if logging is working"""
    print("🔍 Testing logging setup...")
    
    # Test basic print statements
    print("✅ Print statements work")
    
    # Test basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("✅ Basic logging works")
    logger.warning("⚠️ Warning level logging works")
    logger.error("❌ Error level logging works")
    
    # Test if logs go to a file
    log_files = [
        '/app/logs/scanner.log',
        '/var/log/scanner.log',
        './scanner.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"📁 Found log file: {log_file}")
            # Show last few lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"   Last 3 lines:")
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Error reading log: {e}")

def test_scanner_initialization():
    """Test if scanner initializes and runs"""
    print("\n🔧 Testing scanner initialization...")
    
    try:
        from core.scanner import IntelligentForexScanner
        from core.database import DatabaseManager
        import config
        
        print("✅ Imports successful")
        
        # Test database connection
        db_manager = DatabaseManager(config.DATABASE_URL)
        print("✅ Database manager created")
        
        # Test scanner creation
        scanner = IntelligentForexScanner(
            db_manager=db_manager,
            epic_list=getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP'])[:1],  # Just one pair for testing
            claude_api_key=None,  # Disable Claude for testing
            enable_claude_analysis=False
        )
        print("✅ Scanner created")
        
        # Test manual scan
        print("🔍 Testing manual scan...")
        signals = scanner.scan_once()
        print(f"✅ Manual scan completed: {len(signals)} signals")
        
        return scanner
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_schedule_execution():
    """Test if schedule is actually executing"""
    print("\n⏰ Testing schedule execution...")
    
    import schedule
    
    call_count = 0
    
    def test_job():
        nonlocal call_count
        call_count += 1
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"🔄 Schedule executed #{call_count} at {current_time}")
        return call_count
    
    # Schedule every 5 seconds for testing
    schedule.every(5).seconds.do(test_job)
    
    print("⏳ Running schedule test for 20 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 20:
        schedule.run_pending()
        time.sleep(1)
    
    print(f"✅ Schedule test completed. Job ran {call_count} times")
    
    if call_count == 0:
        print("❌ Schedule never executed - there might be an issue with the schedule loop")
    elif call_count < 3:
        print("⚠️ Schedule executed fewer times than expected")
    else:
        print("✅ Schedule is working correctly")

def check_stdout_stderr():
    """Check where stdout/stderr are going"""
    print("\n📤 Checking output streams...")
    
    print(f"stdout: {sys.stdout}")
    print(f"stderr: {sys.stderr}")
    
    # Check if output is being redirected
    if hasattr(sys.stdout, 'name'):
        print(f"stdout name: {sys.stdout.name}")
    if hasattr(sys.stderr, 'name'):
        print(f"stderr name: {sys.stderr.name}")
    
    # Force flush
    sys.stdout.flush()
    sys.stderr.flush()

def check_process_info():
    """Check process and environment info"""
    print(f"\n🔍 Process info:")
    print(f"   PID: {os.getpid()}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Python executable: {sys.executable}")
    
    # Check if running in container
    if os.path.exists('/.dockerenv'):
        print("   ✅ Running in Docker container")
    else:
        print("   ⚠️ Not running in Docker container")

def create_verbose_test_script():
    """Create a test script that shows activity"""
    test_script = '''#!/usr/bin/env python3
import time
import sys
from datetime import datetime

print("🚀 VERBOSE TEST SCRIPT STARTING")
sys.stdout.flush()

for i in range(10):
    current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"📊 [{current_time}] Test iteration #{i+1}")
    sys.stdout.flush()  # Force output immediately
    time.sleep(2)

print("🏁 VERBOSE TEST SCRIPT COMPLETE")
sys.stdout.flush()
'''
    
    with open('/app/verbose_test.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created /app/verbose_test.py")
    print("   Run it with: python /app/verbose_test.py")

if __name__ == "__main__":
    print("🔍 LOGGING AND SCANNER DEBUG")
    print("=" * 50)
    
    check_process_info()
    check_stdout_stderr()
    test_logging()
    
    scanner = test_scanner_initialization()
    
    if scanner:
        test_schedule_execution()
    
    create_verbose_test_script()
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG COMPLETE")
    print("\nNext steps:")
    print("1. Run: python /app/verbose_test.py")
    print("2. Check if you see real-time output")
    print("3. If no output, check container logs with: docker logs <container_name>")