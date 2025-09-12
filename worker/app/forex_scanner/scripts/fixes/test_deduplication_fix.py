#!/usr/bin/env python3
"""
Test script to verify deduplication is working
"""

import sys
sys.path.append('/app/forex_scanner')

def test_deduplication():
    """Test the deduplication system"""
    try:
        from core.database import DatabaseManager
        from alerts.alert_history import AlertHistoryManager
        from alerts.dedup_wrapper import SimpleDeduplicationWrapper
        import config
        
        print("🧪 Testing Deduplication System")
        print("=" * 40)
        
        # Initialize components
        db = DatabaseManager(config.DATABASE_URL)
        alert_history = AlertHistoryManager(db)
        dedup_wrapper = SimpleDeduplicationWrapper(db)
        
        # Test signal
        test_signal = {
            'epic': 'CS.D.USDCHF.MINI.IP',
            'signal_type': 'BULL',
            'strategy': 'test_dedup',
            'confidence_score': 0.75,
            'price': 0.8850,
            'timestamp': '2025-07-14 19:15:00'
        }
        
        print("\n🔍 Testing duplicate detection...")
        
        # First signal should be allowed
        alert_id1 = dedup_wrapper.save_alert_with_dedup(alert_history, test_signal)
        print(f"First signal result: {'✅ Saved' if alert_id1 else '❌ Blocked'} (ID: {alert_id1})")
        
        # Immediate duplicate should be blocked
        alert_id2 = dedup_wrapper.save_alert_with_dedup(alert_history, test_signal)
        print(f"Duplicate signal result: {'❌ Should have been blocked!' if alert_id2 else '✅ Correctly blocked'}")
        
        if alert_id1 and not alert_id2:
            print("\n🎉 Deduplication is working correctly!")
        else:
            print("\n⚠️ Deduplication may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deduplication()
