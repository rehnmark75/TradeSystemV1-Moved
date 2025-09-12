
import sys
import os

print("🧪 MINIMAL TEST START")

# Check if forex_scanner exists
if os.path.exists('/app/forex_scanner'):
    print("✅ forex_scanner directory exists")
else:
    print("❌ forex_scanner directory missing")
    exit(1)

# Add to path
sys.path.insert(0, '/app/forex_scanner')
print("✅ Added to sys.path")

# Try basic import
try:
    print("Attempting: import core")
    import core
    print("✅ import core - SUCCESS")
    
    print("Attempting: from core import scanner")
    from core import scanner
    print("✅ from core import scanner - SUCCESS")
    
    print("Attempting: from core.scanner import ForexScanner")
    from core.scanner import ForexScanner
    print("✅ from core.scanner import ForexScanner - SUCCESS")
    
    print("🎉 ALL IMPORTS SUCCESSFUL!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("🧪 MINIMAL TEST END")
