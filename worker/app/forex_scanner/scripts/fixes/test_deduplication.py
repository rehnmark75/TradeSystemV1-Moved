#!/usr/bin/env python3
"""
Test script for alert deduplication system
"""

import sys
sys.path.append('/app/forex_scanner')

def test_deduplication():
    """Test the alert deduplication system"""
    try:
        from core.database import DatabaseManager
        from core.alert_deduplication import AlertDeduplicationManager
        from alerts.alert_history import AlertHistoryManager
        import config
        
        print("🧪 Testing Alert Deduplication System")
        print("=" * 50)
        
        # Initialize managers
        db = DatabaseManager(config.DATABASE_URL)
        alert_history = AlertHistoryManager(db)
        dedup_manager = AlertDeduplicationManager(db)
        
        print("✅ Managers initialized successfully")
        
        # Test signal
        test_signal = {
            'epic': 'CS.D.EURUSD.CEEM.IP',
            'signal_type': 'BULL',
            'confidence_score': 0.75,
            'strategy': 'test_deduplication',
            'price': 1.0850,
            'timestamp': '2025-07-14 10:00:00'
        }
        
        print("\n🔍 Testing duplicate detection...")
        
        # First signal should be allowed
        allow1, reason1, meta1 = dedup_manager.should_allow_alert(test_signal)
        print(f"First signal: {allow1} - {reason1}")
        
        # Immediate duplicate should be blocked
        allow2, reason2, meta2 = dedup_manager.should_allow_alert(test_signal)
        print(f"Duplicate signal: {allow2} - {reason2}")
        
        if allow1 and not allow2:
            print("✅ Deduplication working correctly!")
        else:
            print("❌ Deduplication not working as expected")
        
        # Test saving with deduplication
        print("\n💾 Testing alert saving with deduplication...")
        alert_id = dedup_manager.save_alert_with_deduplication(
            alert_history, test_signal, "Test deduplication alert"
        )
        
        if alert_id:
            print(f"✅ Alert saved with ID: {alert_id}")
        else:
            print("❌ Alert was not saved (possibly blocked)")
        
        # Get stats
        print("\n📊 Deduplication Statistics:")
        stats = dedup_manager.get_deduplication_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 Deduplication test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deduplication()
