# manual_debug_check_close_deal.py
"""
Updated manual debug check to verify the close deal ID fix
Tests both deal_id (open) and activity_close_deal_id (close) columns
"""

def manual_debug_close_deal():
    """Manual debug using raw connection - testing close deal ID fix"""
    print("🔍 MANUAL DEBUG: Trade Correlation Check - CLOSE DEAL ID FIX")
    print("=" * 70)
    
    try:
        # Use your existing database pattern
        import psycopg2
        import os
        from urllib.parse import urlparse
        
        # Get database URL from environment or config
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            try:
                import config
                database_url = config.DATABASE_URL
            except:
                print("❌ Could not get database URL")
                return
        
        # Parse the URL
        url = urlparse(database_url)
        
        # Connect directly
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            database=url.path[1:],
            user=url.username,
            password=url.password
        )
        
        print("✅ Direct database connection established")
        
        with conn.cursor() as cursor:
            # STEP 1: Compare open vs close deal IDs
            print("\n📋 STEP 1: Comparing Open vs Close Deal IDs")
            print("-" * 50)
            
            cursor.execute("""
                SELECT id, deal_id, activity_close_deal_id, symbol, timestamp, profit_loss, status
                FROM trade_log 
                WHERE (deal_id IS NOT NULL AND deal_id != '') 
                OR (activity_close_deal_id IS NOT NULL AND activity_close_deal_id != '')
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            trades = cursor.fetchall()
            print(f"Found {len(trades)} trades with deal IDs:")
            
            # Analyze both open and close deal IDs
            open_references = []
            close_references = []
            
            for trade in trades:
                trade_id, open_deal_id, close_deal_id, symbol, timestamp, profit_loss, status = trade
                
                print(f"\n  Trade {trade_id} ({symbol}):")
                print(f"    Timestamp: {timestamp}")
                print(f"    Status: {status}")
                print(f"    Current P&L: {profit_loss}")
                print(f"    Open Deal ID:  '{open_deal_id}'")
                print(f"    Close Deal ID: '{close_deal_id}'")
                
                # Extract references from both (same logic as service)
                prefix = "DIAAAAU"
                
                # Extract from open deal ID
                if open_deal_id and open_deal_id.startswith(prefix):
                    open_ref = open_deal_id[len(prefix):]
                else:
                    open_ref = open_deal_id if open_deal_id else None
                
                # Extract from close deal ID
                if close_deal_id and close_deal_id.startswith(prefix):
                    close_ref = close_deal_id[len(prefix):]
                else:
                    close_ref = close_deal_id if close_deal_id else None
                
                if open_ref:
                    open_references.append(open_ref)
                    print(f"    Open Reference:  '{open_ref}'")
                
                if close_ref:
                    close_references.append(close_ref)
                    print(f"    Close Reference: '{close_ref}'")
                
                if not close_ref:
                    print(f"    ⚠️ No close deal ID available")
            
            # STEP 2: Check broker_transactions
            print(f"\n📊 STEP 2: Available Broker Transactions")
            print("-" * 50)
            
            try:
                cursor.execute("""
                    SELECT reference, profit_loss_amount, transaction_date, created_at
                    FROM broker_transactions 
                    ORDER BY created_at DESC 
                    LIMIT 15
                """)
                
                broker_transactions = cursor.fetchall()
                
                if broker_transactions:
                    print(f"Found {len(broker_transactions)} broker transactions:")
                    
                    available_refs = []
                    for tx in broker_transactions:
                        ref, pnl, date, created = tx
                        available_refs.append(ref)
                        print(f"  Reference: '{ref}' (P&L: {pnl}, Date: {date})")
                    
                    # STEP 3: Test matching with OPEN deal IDs (old method)
                    print(f"\n🔍 STEP 3: Testing OPEN Deal ID Matching (Old Method)")
                    print("-" * 50)
                    
                    open_matches = 0
                    for ref_to_find in open_references:
                        if ref_to_find in available_refs:
                            print(f"  ✅ OPEN MATCH: '{ref_to_find}'")
                            open_matches += 1
                        else:
                            print(f"  ❌ NO OPEN MATCH: '{ref_to_find}'")
                    
                    print(f"  📊 Open Deal ID Matches: {open_matches}/{len(open_references)}")
                    
                    # STEP 4: Test matching with CLOSE deal IDs (new method)
                    print(f"\n🎯 STEP 4: Testing CLOSE Deal ID Matching (New Method)")
                    print("-" * 50)
                    
                    close_matches = 0
                    close_match_details = []
                    
                    for ref_to_find in close_references:
                        if ref_to_find in available_refs:
                            print(f"  ✅ CLOSE MATCH: '{ref_to_find}'")
                            close_matches += 1
                            
                            # Find the matching transaction details
                            for tx in broker_transactions:
                                if tx[0] == ref_to_find:
                                    close_match_details.append({
                                        'reference': ref_to_find,
                                        'pnl': tx[1],
                                        'date': tx[2]
                                    })
                                    print(f"      → P&L: {tx[1]}, Date: {tx[2]}")
                                    break
                        else:
                            print(f"  ❌ NO CLOSE MATCH: '{ref_to_find}'")
                            
                            # Check for partial matches
                            partial = [r for r in available_refs[:10] if ref_to_find in r or r in ref_to_find]
                            if partial:
                                print(f"      Similar: {partial[:3]}")
                    
                    print(f"  📊 Close Deal ID Matches: {close_matches}/{len(close_references)}")
                    
                    # STEP 5: Comparison and recommendations
                    print(f"\n📈 STEP 5: Fix Effectiveness Analysis")
                    print("-" * 50)
                    
                    if len(open_references) > 0:
                        open_rate = (open_matches / len(open_references)) * 100
                        print(f"  Open Deal ID Match Rate:  {open_rate:.1f}% ({open_matches}/{len(open_references)})")
                    
                    if len(close_references) > 0:
                        close_rate = (close_matches / len(close_references)) * 100
                        print(f"  Close Deal ID Match Rate: {close_rate:.1f}% ({close_matches}/{len(close_references)})")
                        
                        if close_rate > 0:
                            print(f"\n  🎉 SUCCESS! Close Deal ID matching works!")
                            print(f"  💰 Trades that would get P&L updates:")
                            for match in close_match_details:
                                print(f"      {match['reference']} → {match['pnl']} ({match['date']})")
                        else:
                            print(f"\n  ⚠️ Close Deal ID matching still not working")
                    else:
                        print(f"  ❌ No close deal IDs found in trade_log")
                        print(f"  → Check if activity correlation has run")
                        print(f"  → Check if activity_close_deal_id column is populated")
                    
                    # STEP 6: Next steps recommendations
                    print(f"\n🔧 STEP 6: Recommendations")
                    print("-" * 50)
                    
                    if len(close_references) == 0:
                        print("  1. ❌ No close deal IDs available")
                        print("     → Run activity correlation first to populate activity_close_deal_id")
                        print("     → Check if trades are properly closed")
                    elif close_matches == 0:
                        print("  2. ❌ Close deal IDs exist but no matches")
                        print("     → Check date range - transactions may be from different period")
                        print("     → Verify close deal ID format")
                    else:
                        print(f"  3. ✅ Close deal ID matching works!")
                        print(f"     → Deploy the fixed trade_pnl_correlator.py")
                        print(f"     → Expected match rate: {close_rate:.1f}%")
                        print(f"     → Should update {close_matches} trades with P&L")
                
                else:
                    print("❌ No broker transactions found")
                    print("  → Check if transaction fetching is working")
                    
            except Exception as e:
                print(f"❌ Error checking broker_transactions: {e}")
                print("  → Table may not exist or have different structure")
            
            # STEP 7: Check activity correlation status
            print(f"\n📅 STEP 7: Activity Correlation Status")
            print("-" * 50)
            
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN activity_close_deal_id IS NOT NULL AND activity_close_deal_id != '' THEN 1 END) as with_close_deal_id,
                        COUNT(CASE WHEN activity_correlated = true THEN 1 END) as activity_correlated,
                        COUNT(CASE WHEN profit_loss IS NOT NULL THEN 1 END) as with_pnl
                    FROM trade_log 
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                """)
                
                stats = cursor.fetchone()
                total, with_close, correlated, with_pnl = stats
                
                print(f"  Last 7 days statistics:")
                print(f"    Total trades: {total}")
                print(f"    With close deal ID: {with_close} ({(with_close/total*100) if total > 0 else 0:.1f}%)")
                print(f"    Activity correlated: {correlated} ({(correlated/total*100) if total > 0 else 0:.1f}%)")
                print(f"    With P&L data: {with_pnl} ({(with_pnl/total*100) if total > 0 else 0:.1f}%)")
                
                if with_close == 0:
                    print(f"\n  ⚠️ No trades have close deal IDs yet")
                    print(f"     → Activity correlation needs to run first")
                elif with_close < total * 0.8:
                    print(f"\n  ⚠️ Low close deal ID coverage")
                    print(f"     → Some trades may not be properly closed")
                else:
                    print(f"\n  ✅ Good close deal ID coverage")
                
            except Exception as e:
                print(f"❌ Error checking activity correlation: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    manual_debug_close_deal()