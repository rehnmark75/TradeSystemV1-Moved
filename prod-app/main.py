from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from routers.orders_router import router as orders_router
import logging
import os

os.makedirs("/app/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/uvicorn-prod.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Global middleware (if needed)
@app.middleware("http")
async def require_verified_gateway(request, call_next):
    if request.headers.get("x-apim-gateway") != "verified":
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    return await call_next(request)

# Optional root block route
@app.get("/")
@app.post("/")
def block_root():
    raise HTTPException(status_code=403, detail="****")

@app.get("/favicon.ico")
async def ignore_favicon():
    return Response(status_code=204)

@app.get("/validate")
def validate():
    return PlainTextResponse("API is working")

# Register routers
app.include_router(orders_router, prefix="/orders", tags=["orders"])