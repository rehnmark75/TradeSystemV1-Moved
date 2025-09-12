#!/usr/bin/env python3
"""
Claude Database Verifier
Checks if Claude analysis is properly saved to database
"""

import sys
sys.path.insert(0, '/app/forex_scanner')

def verify_claude_database_integration():
    """Verify Claude data in database"""
    
    try:
        from core.database import DatabaseManager
        import config
        
        db = DatabaseManager(config.DATABASE_URL)
        
        print("🔍 VERIFYING CLAUDE DATABASE INTEGRATION")
        print("=" * 50)
        
        # Check for Claude columns
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'alert_history' 
            AND column_name LIKE 'claude_%'
        """)
        
        claude_columns = [row[0] for row in cursor.fetchall()]
        print(f"✅ Claude columns found: {claude_columns}")
        
        # Check for Claude data
        cursor.execute("""
            SELECT COUNT(*) as total,
                   COUNT(claude_score) as with_score,
                   COUNT(claude_decision) as with_decision,
                   COUNT(claude_approved) as with_approved
            FROM alert_history
        """)
        
        stats = cursor.fetchone()
        total, with_score, with_decision, with_approved = stats
        
        print(f"📊 Alert Statistics:")
        print(f"   Total alerts: {total}")
        print(f"   With Claude score: {with_score}")
        print(f"   With Claude decision: {with_decision}")
        print(f"   With Claude approval: {with_approved}")
        
        # Show recent Claude data
        cursor.execute("""
            SELECT id, epic, claude_score, claude_decision, claude_approved, 
                   substring(claude_reason, 1, 50) as reason_preview
            FROM alert_history 
            WHERE claude_score IS NOT NULL
            ORDER BY alert_timestamp DESC 
            LIMIT 5
        """)
        
        recent_claude = cursor.fetchall()
        
        if recent_claude:
            print(f"\n📋 Recent Claude Analysis:")
            print("   ID | Epic | Score | Decision | Approved | Reason")
            print("   " + "-" * 50)
            for row in recent_claude:
                print(f"   {row[0]} | {row[1]} | {row[2]}/10 | {row[3]} | {row[4]} | {row[5]}...")
        else:
            print("\n❌ No Claude analysis found in database")
        
        cursor.close()
        conn.close()
        
        return with_score > 0
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_claude_database_integration()
