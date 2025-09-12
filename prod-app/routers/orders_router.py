from fastapi import APIRouter, Request, HTTPException, Depends
from datetime import datetime
import json
import traceback
from services.ig_auth import ig_login
from services.ig_orders import has_open_position, place_market_order
from services.ig_market import get_current_bid_price, get_last_15m_candle_range
from services.ig_risk_utils import get_atr, calculate_dynamic_sl_tp


from services.keyvault import get_secret
from config import EPIC_MAP, API_BASE_URL
from dependencies import get_ig_auth_headers
import logging

router = APIRouter()

# Configure logger
logger = logging.getLogger("tradelogger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/prodtrade.log")
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@router.post("/place-order")
async def ig_place_order(request: Request, trading_headers: dict = Depends(get_ig_auth_headers)):
    try:
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8").strip()

        if not body_str:
            raise HTTPException(status_code=400, detail="Request body is empty")

        if "," not in body_str:
            raise HTTPException(status_code=400, detail="Invalid format, expected 'EPIC,DIRECTION'")

        try:
            epic, direction = map(lambda x: x.strip().upper().replace('"', ''), body_str.split(",", 1))
        except ValueError:
            raise HTTPException(status_code=400, detail="Unable to parse EPIC and DIRECTION")

        if not epic or not direction:
            raise HTTPException(status_code=400, detail="EPIC and DIRECTION cannot be empty")

        print(f"Parsed EPIC: {epic}, Direction: {direction}")

        symbol = EPIC_MAP.get(epic.upper())
        if not symbol:
            raise HTTPException(status_code=404, detail=f"No mapping found for epic: {epic}")

        if await has_open_position(symbol, trading_headers):
            msg = f"Position already open for {symbol}, skipping order."
            print(msg)
            logger.info(json.dumps(msg))
            return {"status": "skipped", "message": msg}

        print(f"No open position for {symbol}, placing order.")
        logger.info(json.dumps(f"No open position for {symbol}, placing order."))

        price_info = await get_current_bid_price(trading_headers, symbol)
        currency = price_info["currency"]
        currency_code = price_info["currency_code"]
        
        # TODO: Dynamic SL/TP logic here if needed
        test = await get_last_15m_candle_range(trading_headers, symbol)

        # get stop loss
        # Step X: Calculate ATR
        atr = await get_atr(symbol, trading_headers)

        # Step Y: Compute stop/limit distances
        sl_tp = await calculate_dynamic_sl_tp(symbol, trading_headers, atr, rr_ratio=2.0)

        stop_distance = sl_tp["stopDistance"]
        limit_distance = sl_tp["limitDistance"]
        print(stop_distance)
        print(limit_distance)
        

        # result = await place_market_order(trading_headers, symbol, direction, currency_code)
        # result["timestamp"] = datetime.utcnow().isoformat()
        # logger.info(json.dumps(result))

        return {
            "status": "success",
            # "dealReference": result.get("dealReference")
        }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as ex:
        print("EXCEPTION:", repr(ex))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")