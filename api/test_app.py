#!/usr/bin/env python3
"""
Test FastAPI Application for TradingView Scripts API

Simple test application to validate the TradingView API router functionality.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the TradingView API router
from api.tvscripts_api import router as tvscripts_router

# Create FastAPI app
app = FastAPI(
    title="TradingView Scripts API",
    description="API for searching, analyzing, and importing TradingView strategies",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the TradingView router
app.include_router(tvscripts_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TradingView Scripts API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/tvscripts/health",
        "endpoints": {
            "search": "POST /api/tvscripts/search",
            "get_script": "GET /api/tvscripts/script/{slug}",
            "analyze": "POST /api/tvscripts/analyze",
            "import": "POST /api/tvscripts/import",
            "list_imports": "GET /api/tvscripts/imports",
            "stats": "GET /api/tvscripts/stats",
            "validate": "GET /api/tvscripts/validate"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy", "service": "tradingview-api"}

if __name__ == "__main__":
    # Run the test server
    print("🚀 Starting TradingView Scripts API test server...")
    print("📖 API Documentation: http://localhost:8080/docs")
    print("🔍 Health Check: http://localhost:8080/api/tvscripts/health")
    
    uvicorn.run(
        "test_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )