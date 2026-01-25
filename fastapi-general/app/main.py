import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.scanners import ScannerManager


class ClaudeAnalyzeRequest(BaseModel):
    ticker: Optional[str] = None
    signal_id: Optional[int] = None
    analysis_level: str = "standard"
    model: Optional[str] = None


app = FastAPI(title="fastapi-general")


@app.on_event("startup")
async def startup() -> None:
    db_url = os.environ.get("STOCKS_DATABASE_URL")
    if not db_url:
        raise RuntimeError("STOCKS_DATABASE_URL is required")
    db = AsyncDatabaseManager(db_url)
    await db.connect()
    manager = ScannerManager(db)
    await manager.initialize()
    app.state.db = db
    app.state.manager = manager


@app.on_event("shutdown")
async def shutdown() -> None:
    db = getattr(app.state, "db", None)
    if db:
        await db.close()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/claude/analyze")
async def claude_analyze(payload: ClaudeAnalyzeRequest) -> dict:
    if not payload.ticker and not payload.signal_id:
        raise HTTPException(status_code=400, detail="ticker or signal_id is required")

    db = app.state.db
    manager: ScannerManager = app.state.manager

    signal_id = payload.signal_id
    ticker = payload.ticker

    if not signal_id and ticker:
        row = await db.fetchrow(
            """
            SELECT id
            FROM stock_scanner_signals
            WHERE ticker = $1
            ORDER BY signal_timestamp DESC
            LIMIT 1
            """,
            ticker.upper(),
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"No signal found for {ticker}")
        signal_id = row["id"]

    analysis = await manager.analyze_single_signal_with_claude(
        signal_id=signal_id,
        analysis_level=payload.analysis_level,
        model=payload.model,
    )

    if not analysis:
        raise HTTPException(status_code=500, detail="Claude analysis failed")

    return {
        "status": "ok",
        "signal_id": signal_id,
        "ticker": ticker,
        "analysis": analysis.to_dict(),
    }
