import httpx

API_BASE_URL = "https://demo-api.ig.com/gateway/deal"

async def modify_stop_loss(deal_id, stop_level, headers):
    print(f"🔧 Updating SL for {deal_id} to {stop_level}")

    payload = {"stopLevel": stop_level}
    ig_headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json; charset=UTF-8",
        "Version": "2",
        "X-IG-API-KEY": headers.get("X-IG-API-KEY"),
        "CST": headers.get("CST"),
        "X-SECURITY-TOKEN": headers.get("X-SECURITY-TOKEN")
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(f"{API_BASE_URL}/positions/otc/{deal_id}", json=payload, headers=ig_headers)
            if response.status_code == 200:
                print(f"✅ Stop loss updated for deal {deal_id}")
                return True
            else:
                print(f"❌ Failed to update SL: {response.status_code} | {response.text}")
                return False
    except Exception as e:
        print(f"🚨 Exception during SL update: {e}")
        return False
