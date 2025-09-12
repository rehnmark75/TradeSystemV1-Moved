#!/usr/bin/env python3
"""
Close Price Corrector - Fix the systematic +8 pip error in historical data
Focus on close prices since they're most critical for strategy calculations
"""

import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from services.db import SessionLocal
from services.models import IGCandle
from igstream.ig_auth_prod import ig_login
from services.keyvault import get_secret

logger = logging.getLogger(__name__)

class ClosePriceCorrector:
    """Fix corrupted close prices with accurate API data"""
    
    def __init__(self):
        self.headers = {}
        self.api_base = "https://api.ig.com/gateway/deal"
    
    async def authenticate(self):
        """Authenticate with IG API"""
        try:
            api_key = get_secret("prodapikey")
            password = get_secret("prodpwd") 
            username = "rehnmarkh"
            
            auth_result = await ig_login(api_key, password, username)
            self.headers = {
                "CST": auth_result["CST"],
                "X-SECURITY-TOKEN": auth_result["X-SECURITY-TOKEN"],
                "VERSION": "3",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-IG-API-KEY": api_key
            }
            logger.info("✅ Authenticated with IG API")
            return True
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    async def test_api_response(self, epic: str = "CS.D.EURUSD.CEEM.IP"):
        """Test what the API actually returns"""
        try:
            async with httpx.AsyncClient(base_url=self.api_base, headers=self.headers) as client:
                # Get recent data to see response structure
                from_time = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
                to_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                
                url = f"/prices/{epic}"
                params = {
                    "resolution": "MINUTE_5",
                    "from": from_time,
                    "to": to_time,
                    "max": 5
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                print("🔍 IG API RESPONSE STRUCTURE:")
                print("=" * 50)
                
                if "prices" in data and len(data["prices"]) > 0:
                    sample_candle = data["prices"][0]
                    print("Sample candle fields:")
                    for key, value in sample_candle.items():
                        print(f"  {key}: {value}")
                    
                    # Check what price fields are available
                    if "closePrice" in sample_candle:
                        close_fields = sample_candle["closePrice"]
                        print(f"\nClose price fields: {close_fields}")
                    
                    return data["prices"]
                else:
                    print("No price data returned")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return None
    
    def extract_close_price(self, candle_data):
        """Extract the most accurate close price from API response"""
        try:
            # Try different possible structures
            if "closePrice" in candle_data:
                close_data = candle_data["closePrice"]
                
                # If it has bid/ask, calculate mid
                if isinstance(close_data, dict) and "bid" in close_data and "ask" in close_data:
                    return (close_data["bid"] + close_data["ask"]) / 2
                
                # If it's just a number
                elif isinstance(close_data, (int, float)):
                    return float(close_data)
            
            # Fallback - try lastTradedPrice
            if "lastTradedPrice" in candle_data:
                return float(candle_data["lastTradedPrice"])
            
            # Last resort - calculate from bid/ask if available
            if "bid" in candle_data and "ask" in candle_data:
                return (candle_data["bid"] + candle_data["ask"]) / 2
            
            return None
            
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Could not extract close price: {e}")
            return None
    
    async def fix_corrupted_period(self, epic: str, start_date: str, end_date: str, timeframe: int = 5):
        """Fix corrupted close prices for a specific period"""
        logger.info(f"🔧 Fixing close prices for {epic} from {start_date} to {end_date}")
        
        try:
            async with httpx.AsyncClient(base_url=self.api_base, headers=self.headers) as client:
                # Get API data for the period
                url = f"/prices/{epic}"
                resolution = "MINUTE_5" if timeframe == 5 else "MINUTE_15" if timeframe == 15 else "HOUR"
                
                params = {
                    "resolution": resolution,
                    "from": f"{start_date}T00:00:00", 
                    "to": f"{end_date}T23:59:59",
                    "max": 1000
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                api_candles = data.get("prices", [])
                
                if not api_candles:
                    logger.warning(f"No API data for {epic} in period {start_date} to {end_date}")
                    return 0
                
                # Update corrupted database entries
                fixed_count = 0
                with SessionLocal() as session:
                    for api_candle in api_candles:
                        # Parse timestamp
                        timestamp_str = api_candle.get("snapshotTime", "")
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S")
                        except ValueError:
                            continue
                        
                        # Get correct close price from API
                        correct_close = self.extract_close_price(api_candle)
                        if correct_close is None:
                            continue
                        
                        # Find corrupted database entry
                        db_candle = session.query(IGCandle).filter(
                            IGCandle.epic == epic,
                            IGCandle.timeframe == timeframe,
                            IGCandle.start_time == timestamp,
                            IGCandle.data_source == 'chart_streamer'  # Only fix corrupted streaming data
                        ).first()
                        
                        if db_candle:
                            old_close = db_candle.close
                            difference = abs(old_close - correct_close) * 10000  # pips
                            
                            # Only update if there's a significant difference (>2 pips)
                            if difference > 2:
                                db_candle.close = correct_close
                                # Also update other OHLC if available (but close is most important)
                                if "openPrice" in api_candle:
                                    open_price = self.extract_price_field(api_candle, "openPrice")
                                    if open_price:
                                        db_candle.open = open_price
                                
                                db_candle.data_source = "close_price_corrected"
                                db_candle.updated_at = datetime.now()
                                
                                fixed_count += 1
                                logger.info(f"Fixed {timestamp}: {old_close:.5f} -> {correct_close:.5f} ({difference:.1f} pip correction)")
                    
                    session.commit()
                
                logger.info(f"✅ Fixed {fixed_count} close prices for {epic}")
                return fixed_count
                
        except Exception as e:
            logger.error(f"❌ Failed to fix period {start_date}-{end_date}: {e}")
            return 0
    
    def extract_price_field(self, candle_data, field_name):
        """Extract any price field (open, high, low, close) from API response"""
        try:
            if field_name in candle_data:
                price_data = candle_data[field_name]
                if isinstance(price_data, dict) and "bid" in price_data and "ask" in price_data:
                    return (price_data["bid"] + price_data["ask"]) / 2
                elif isinstance(price_data, (int, float)):
                    return float(price_data)
            return None
        except:
            return None

async def main():
    """Test the API structure and fix critical timestamps"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    corrector = ClosePriceCorrector()
    
    if not await corrector.authenticate():
        return
    
    print("🧪 Testing IG API response structure...")
    sample_data = await corrector.test_api_response()
    
    if sample_data:
        print("\n🎯 Ready to fix corrupted data!")
        print("Focus: Close prices (most critical for Zero Lag EMA)")
        
        # Fix the specific problematic period we discovered
        print(f"\n🔧 Fixing critical period: 2025-09-01 (the 15.3 pip discrepancy)")
        fixed = await corrector.fix_corrupted_period(
            epic="CS.D.EURUSD.CEEM.IP",
            start_date="2025-09-01", 
            end_date="2025-09-01",
            timeframe=5
        )
        
        if fixed > 0:
            print(f"✅ Successfully fixed {fixed} corrupted close prices!")
            print("💡 You can now run your Zero Lag strategy with accurate data")
        else:
            print("ℹ️ No corrupted data found in this period")

if __name__ == "__main__":
    asyncio.run(main())