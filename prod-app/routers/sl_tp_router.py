from fastapi import APIRouter, Request, HTTPException, Query, Depends
from services.ig_risk_utils import get_atr, calculate_dynamic_sl_tp
from dependencies import get_ig_auth_headers

router = APIRouter()

@router.get("/sl-tp")
async def get_sl_tp(
    request: Request,
    epic: str = Query(..., description="IG epic (symbol) to calculate SL/TP for"),
    rr_ratio: float = Query(2.0, description="Risk/reward ratio for TP vs SL"),
    trading_headers: dict = Depends(get_ig_auth_headers)
):
    try:
        

        atr = await get_atr(epic, trading_headers)
        sl_tp = await calculate_dynamic_sl_tp(epic, trading_headers, atr, rr_ratio=rr_ratio)

        return {
            "epic": epic,
            "atr": atr,
            "stopDistance": sl_tp["stopDistance"],
            "limitDistance": sl_tp["limitDistance"]
        }

    except Exception as ex:
        import traceback
        print("EXCEPTION:", repr(ex))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to calculate SL/TP")
