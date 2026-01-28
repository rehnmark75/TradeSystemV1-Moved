import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.scanners import ScannerManager
from stock_scanner.services import StockClaudeAnalyzer


class ClaudeAnalyzeRequest(BaseModel):
    ticker: Optional[str] = None
    signal_id: Optional[int] = None
    analysis_level: str = "standard"
    model: Optional[str] = None


class ClaudeWatchlistAnalyzeRequest(BaseModel):
    watchlist_name: str
    ticker: str
    scan_date: Optional[str] = None
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


@app.post("/claude/analyze-watchlist")
async def claude_analyze_watchlist(payload: ClaudeWatchlistAnalyzeRequest) -> dict:
    if not payload.watchlist_name or not payload.ticker:
        raise HTTPException(status_code=400, detail="watchlist_name and ticker are required")

    db = app.state.db
    manager: ScannerManager = app.state.manager

    watchlist_name = payload.watchlist_name
    ticker = payload.ticker.upper()

    params = [watchlist_name, ticker]
    date_clause = ""
    if payload.scan_date:
        date_clause = "AND w.scan_date = $3"
        params.append(payload.scan_date)

    query = f"""
        SELECT
            w.watchlist_name,
            w.ticker,
            w.scan_date,
            w.crossover_date,
            w.price,
            w.volume,
            w.avg_volume,
            w.rsi_14,
            w.macd,
            w.macd_histogram,
            w.gap_pct,
            w.price_change_1d,
            w.avg_daily_change_5d,
            w.daq_score,
            w.daq_grade,
            w.atr_14,
            w.atr_percent,
            w.swing_high,
            w.swing_low,
            w.suggested_entry_low,
            w.suggested_entry_high,
            w.suggested_stop_loss,
            w.suggested_target_1,
            w.suggested_target_2,
            w.risk_reward_ratio,
            w.risk_percent,
            w.volume_trend,
            w.relative_volume,
            w.rs_percentile,
            w.rs_trend
        FROM stock_watchlist_results w
        WHERE w.watchlist_name = $1
          AND w.ticker = $2
          {date_clause}
        ORDER BY w.scan_date DESC NULLS LAST
        LIMIT 1
    """

    row = await db.fetchrow(query, *params)
    if not row and payload.scan_date:
        fallback_params = [watchlist_name, ticker]
        fallback_query = """
            SELECT
                w.watchlist_name,
                w.ticker,
                w.scan_date,
                w.crossover_date,
                w.price,
                w.volume,
                w.avg_volume,
                w.rsi_14,
                w.macd,
                w.macd_histogram,
                w.gap_pct,
                w.price_change_1d,
                w.avg_daily_change_5d,
                w.daq_score,
                w.daq_grade,
                w.atr_14,
                w.atr_percent,
                w.swing_high,
                w.swing_low,
                w.suggested_entry_low,
                w.suggested_entry_high,
                w.suggested_stop_loss,
                w.suggested_target_1,
                w.suggested_target_2,
                w.risk_reward_ratio,
                w.risk_percent,
                w.volume_trend,
                w.relative_volume,
                w.rs_percentile,
                w.rs_trend
            FROM stock_watchlist_results w
            WHERE w.watchlist_name = $1
              AND w.ticker = $2
            ORDER BY w.scan_date DESC NULLS LAST
            LIMIT 1
        """
        row = await db.fetchrow(fallback_query, *fallback_params)

    if not row:
        raise HTTPException(status_code=404, detail=f"No watchlist entry found for {ticker}")

    watch = dict(row)
    price = watch.get("price") or 0
    entry_price = watch.get("suggested_entry_low") or watch.get("suggested_entry_high") or price
    stop_loss = watch.get("suggested_stop_loss") or (price * 0.95 if price else None)
    take_profit_1 = watch.get("suggested_target_1") or watch.get("suggested_target_2") or (price * 1.05 if price else None)

    confluence = [
        f"watchlist:{watchlist_name}",
        watch.get("volume_trend"),
        watch.get("rs_trend")
    ]
    confluence = [c for c in confluence if c]

    signal: Dict[str, Any] = {
        "ticker": ticker,
        "signal_type": "BUY",
        "scanner_name": f"watchlist:{watchlist_name}",
        "composite_score": watch.get("daq_score") or 0,
        "quality_tier": watch.get("daq_grade") or "C",
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": watch.get("suggested_target_2"),
        "risk_reward_ratio": watch.get("risk_reward_ratio") or 1.5,
        "risk_percent": watch.get("risk_percent") or None,
        "confluence_factors": confluence,
    }

    technical_data = {
        "rsi_14": watch.get("rsi_14"),
        "macd": watch.get("macd"),
        "macd_histogram": watch.get("macd_histogram"),
        "relative_volume": watch.get("relative_volume"),
        "atr_14": watch.get("atr_14"),
        "atr_percent": watch.get("atr_percent"),
        "swing_high": watch.get("swing_high"),
        "swing_low": watch.get("swing_low"),
        "suggested_entry_low": watch.get("suggested_entry_low"),
        "suggested_entry_high": watch.get("suggested_entry_high"),
        "suggested_stop_loss": watch.get("suggested_stop_loss"),
        "suggested_target_1": watch.get("suggested_target_1"),
        "suggested_target_2": watch.get("suggested_target_2"),
        "risk_reward_ratio": watch.get("risk_reward_ratio"),
        "risk_percent": watch.get("risk_percent"),
        "volume_trend": watch.get("volume_trend"),
        "rs_percentile": watch.get("rs_percentile"),
        "rs_trend": watch.get("rs_trend"),
        "gap_pct": watch.get("gap_pct"),
        "price_change_1d": watch.get("price_change_1d"),
        "avg_daily_change_5d": watch.get("avg_daily_change_5d"),
    }

    technical_data = {k: v for k, v in technical_data.items() if v is not None}

    smc_data = await manager._get_signal_smc_data(signal)
    if smc_data:
        technical_data["smc"] = smc_data

    fundamental_data = await manager._get_signal_fundamental_data(signal)

    analyzer = StockClaudeAnalyzer(
        default_model=payload.model or "sonnet",
        db_manager=db,
        enable_charts=True
    )

    if not analyzer.is_available:
        raise HTTPException(status_code=503, detail="Claude API not available")

    analysis = await analyzer.analyze_signal(
        signal=signal,
        technical_data=technical_data,
        fundamental_data=fundamental_data,
        analysis_level=payload.analysis_level,
        model=payload.model,
        include_chart=True
    )

    if not analysis:
        raise HTTPException(status_code=500, detail="Claude analysis failed")

    analysis_dict = analysis.to_dict()

    await db.execute(
        """
        INSERT INTO stock_watchlist_claude_analysis (
            watchlist_name,
            ticker,
            scan_date,
            claude_grade,
            claude_score,
            claude_conviction,
            claude_action,
            claude_thesis,
            claude_key_strengths,
            claude_key_risks,
            claude_position_rec,
            claude_stop_adjustment,
            claude_time_horizon,
            claude_raw_response,
            claude_analyzed_at,
            claude_tokens_used,
            claude_latency_ms,
            claude_model
        ) VALUES (
            $1, $2, $3,
            $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
        )
        ON CONFLICT (watchlist_name, ticker, scan_date)
        DO UPDATE SET
            claude_grade = EXCLUDED.claude_grade,
            claude_score = EXCLUDED.claude_score,
            claude_conviction = EXCLUDED.claude_conviction,
            claude_action = EXCLUDED.claude_action,
            claude_thesis = EXCLUDED.claude_thesis,
            claude_key_strengths = EXCLUDED.claude_key_strengths,
            claude_key_risks = EXCLUDED.claude_key_risks,
            claude_position_rec = EXCLUDED.claude_position_rec,
            claude_stop_adjustment = EXCLUDED.claude_stop_adjustment,
            claude_time_horizon = EXCLUDED.claude_time_horizon,
            claude_raw_response = EXCLUDED.claude_raw_response,
            claude_analyzed_at = EXCLUDED.claude_analyzed_at,
            claude_tokens_used = EXCLUDED.claude_tokens_used,
            claude_latency_ms = EXCLUDED.claude_latency_ms,
            claude_model = EXCLUDED.claude_model,
            updated_at = NOW()
        """,
        watchlist_name,
        ticker,
        watch.get("scan_date"),
        analysis_dict.get("claude_grade"),
        analysis_dict.get("claude_score"),
        analysis_dict.get("claude_conviction"),
        analysis_dict.get("claude_action"),
        analysis_dict.get("claude_thesis"),
        analysis_dict.get("claude_key_strengths"),
        analysis_dict.get("claude_key_risks"),
        analysis_dict.get("claude_position_rec"),
        analysis_dict.get("claude_stop_adjustment"),
        analysis_dict.get("claude_time_horizon"),
        analysis_dict.get("claude_raw_response"),
        analysis_dict.get("claude_analyzed_at"),
        analysis_dict.get("claude_tokens_used"),
        analysis_dict.get("claude_latency_ms"),
        analysis_dict.get("claude_model"),
    )

    return {
        "status": "ok",
        "ticker": ticker,
        "watchlist_name": watchlist_name,
        "analysis": analysis_dict,
    }
