"""
TradingView Scripts FastAPI Router

Provides REST API endpoints for TradingView script search, analysis,
and strategy import functionality. Integrates with the MCP server
and TradeSystemV1 configuration system.
"""

import sys
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "mcp"))
sys.path.insert(0, str(project_root / "strategy_bridge"))
sys.path.insert(0, str(project_root / "worker" / "app" / "forex_scanner"))

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query for TradingView scripts")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    open_source_only: bool = Field(default=True, description="Only return open-source scripts")
    script_type: Optional[str] = Field(default=None, description="Filter by script type")
    min_likes: Optional[int] = Field(default=None, description="Minimum likes threshold")

class AnalysisRequest(BaseModel):
    """Script analysis request model"""
    slug: Optional[str] = Field(default=None, description="Script slug to analyze")
    code: Optional[str] = Field(default=None, description="Pine Script code to analyze directly")

class ImportRequest(BaseModel):
    """Strategy import request model"""
    slug: str = Field(..., description="Script slug to import")
    target_config: Optional[str] = Field(default=None, description="Target configuration file")
    preset_name: str = Field(default="tradingview", description="Name for the new preset")
    strategy_name: Optional[str] = Field(default=None, description="Custom strategy name")
    run_optimization: bool = Field(default=False, description="Run parameter optimization after import")

class ScriptResponse(BaseModel):
    """Script response model"""
    slug: str
    title: str
    author: str
    description: Optional[str]
    open_source: bool
    url: Optional[str]
    tags: Optional[str]
    likes_count: Optional[int]
    uses_count: Optional[int]
    script_type: Optional[str]

class AnalysisResponse(BaseModel):
    """Analysis response model"""
    analysis_complete: bool
    inputs: List[Dict[str, Any]]
    signals: Dict[str, Any]
    rules: List[Dict[str, Any]]
    code_stats: Dict[str, Any]
    strategy_classification: str
    complexity_score: float

class ImportResponse(BaseModel):
    """Import response model"""
    success: bool
    message: str
    target_config: Optional[str]
    preset_name: Optional[str]
    config_preview: Optional[Dict[str, Any]]
    optimization_scheduled: bool = False
    error: Optional[str]

class DatabaseStatsResponse(BaseModel):
    """Database statistics response model"""
    total_scripts: int
    open_source_scripts: int
    scripts_with_code: int
    script_types: Dict[str, int]
    strategy_imports: int

# Router instance
router = APIRouter(prefix="/api/tvscripts", tags=["tradingview"])

def get_mcp_client():
    """Dependency to get MCP client instance"""
    try:
        from client.mcp_client import TVScriptsClient
        return TVScriptsClient()
    except ImportError:
        raise HTTPException(status_code=503, detail="MCP client not available")

def get_integrator():
    """Dependency to get TradingView integrator"""
    try:
        from configdata.strategies.tradingview_integration import get_integrator
        return get_integrator()
    except ImportError:
        raise HTTPException(status_code=503, detail="TradingView integration not available")

@router.get("/health", summary="Health check")
async def health_check():
    """Check API health and availability of dependencies"""
    try:
        # Test MCP client availability
        mcp_available = False
        try:
            from client.mcp_client import TVScriptsClient
            mcp_available = True
        except ImportError:
            pass
        
        # Test integration availability
        integration_available = False
        try:
            from configdata.strategies.tradingview_integration import validate_tradingview_integration
            validation = validate_tradingview_integration()
            integration_available = validation.get('valid', False)
        except ImportError:
            pass
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mcp_client_available": mcp_available,
            "integration_available": integration_available,
            "api_version": "0.1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/search", response_model=List[ScriptResponse], summary="Search TradingView scripts")
async def search_scripts(request: SearchRequest, client=Depends(get_mcp_client)):
    """
    Search TradingView scripts with filters
    
    Returns a list of scripts matching the search criteria.
    Only open-source scripts with available code are returned by default.
    """
    try:
        filters = {
            "open_source_only": request.open_source_only
        }
        
        if request.script_type:
            filters["script_type"] = request.script_type
        if request.min_likes:
            filters["min_likes"] = request.min_likes
        
        with client:
            results = client.search(request.query, request.limit, filters)
        
        # Convert to response models
        scripts = []
        for result in results:
            script = ScriptResponse(
                slug=result.get("slug", ""),
                title=result.get("title", ""),
                author=result.get("author", ""),
                description=result.get("description"),
                open_source=result.get("open_source", False),
                url=result.get("url"),
                tags=result.get("tags"),
                likes_count=result.get("likes_count"),
                uses_count=result.get("uses_count"),
                script_type=result.get("script_type")
            )
            scripts.append(script)
        
        return scripts
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/script/{slug}", response_model=ScriptResponse, summary="Get script details")
async def get_script(slug: str, client=Depends(get_mcp_client)):
    """
    Get detailed information about a specific script
    
    Returns script metadata and code (if open-source).
    """
    try:
        with client:
            script_data = client.get_script(slug)
        
        if not script_data:
            raise HTTPException(status_code=404, detail=f"Script not found: {slug}")
        
        return ScriptResponse(
            slug=script_data.get("slug", slug),
            title=script_data.get("title", ""),
            author=script_data.get("author", ""),
            description=script_data.get("description"),
            open_source=script_data.get("open_source", False),
            url=script_data.get("url"),
            tags=script_data.get("tags"),
            likes_count=script_data.get("likes_count"),
            uses_count=script_data.get("uses_count"),
            script_type=script_data.get("script_type")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get script failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get script: {str(e)}")

@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze Pine Script")
async def analyze_script(request: AnalysisRequest, client=Depends(get_mcp_client)):
    """
    Analyze Pine Script code to extract patterns and indicators
    
    Provide either a script slug or Pine Script code directly.
    Returns detailed analysis including indicators, signals, and strategy classification.
    """
    try:
        if not request.slug and not request.code:
            raise HTTPException(status_code=400, detail="Either slug or code must be provided")
        
        with client:
            if request.slug:
                analysis = client.analyze_script(slug=request.slug)
            else:
                analysis = client.analyze_script(code=request.code)
        
        if not analysis:
            raise HTTPException(status_code=400, detail="Analysis failed")
        
        signals = analysis.get("signals", {})
        
        return AnalysisResponse(
            analysis_complete=analysis.get("analysis_complete", False),
            inputs=analysis.get("inputs", []),
            signals=signals,
            rules=analysis.get("rules", []),
            code_stats=analysis.get("code_stats", {}),
            strategy_classification=signals.get("strategy_type", "unknown"),
            complexity_score=signals.get("complexity_score", 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/import", response_model=ImportResponse, summary="Import TradingView strategy")
async def import_strategy(
    request: ImportRequest, 
    background_tasks: BackgroundTasks,
    integrator=Depends(get_integrator)
):
    """
    Import a TradingView strategy into TradeSystemV1 configuration
    
    Downloads and analyzes the script, then generates a configuration preset
    for the existing strategy system. Optionally schedules parameter optimization.
    """
    try:
        # Import the strategy
        result = integrator.import_strategy(
            slug=request.slug,
            target_config=request.target_config,
            preset_name=request.preset_name
        )
        
        if not result.get("success"):
            return ImportResponse(
                success=False,
                message="Import failed",
                error=result.get("error", "Unknown error")
            )
        
        response = ImportResponse(
            success=True,
            message=f"Successfully imported strategy as '{request.preset_name}' preset",
            target_config=result.get("target_config"),
            preset_name=request.preset_name,
            config_preview=result.get("config_preview"),
            optimization_scheduled=False
        )
        
        # Schedule optimization if requested
        if request.run_optimization:
            background_tasks.add_task(
                schedule_optimization,
                request.slug,
                result.get("target_config"),
                request.preset_name
            )
            response.optimization_scheduled = True
            response.message += " (optimization scheduled)"
        
        return response
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.get("/imports", summary="List imported strategies")
async def list_imports(integrator=Depends(get_integrator)):
    """
    List all imported TradingView strategies
    
    Returns a list of previously imported strategies with their metadata.
    """
    try:
        imports = integrator.list_imports()
        return {
            "imports": imports,
            "count": len(imports),
            "status": integrator.get_import_status()
        }
        
    except Exception as e:
        logger.error(f"List imports failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list imports: {str(e)}")

@router.delete("/imports/{slug}", summary="Remove imported strategy")
async def remove_import(slug: str, integrator=Depends(get_integrator)):
    """
    Remove an imported strategy (marks as removed)
    
    Note: This doesn't delete the preset from configuration files,
    it only marks the import as removed in the registry.
    """
    try:
        success = integrator.remove_import(slug)
        
        if success:
            return {"message": f"Import {slug} marked as removed"}
        else:
            raise HTTPException(status_code=404, detail=f"Import not found: {slug}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove import: {str(e)}")

@router.get("/stats", response_model=DatabaseStatsResponse, summary="Get database statistics")
async def get_stats(client=Depends(get_mcp_client)):
    """
    Get TradingView scripts database statistics
    
    Returns information about the script database including counts and types.
    """
    try:
        with client:
            stats = client.get_stats()
        
        if not stats:
            raise HTTPException(status_code=500, detail="Failed to get statistics")
        
        return DatabaseStatsResponse(
            total_scripts=stats.get("total_scripts", 0),
            open_source_scripts=stats.get("open_source_scripts", 0),
            scripts_with_code=stats.get("scripts_with_code", 0),
            script_types=stats.get("script_types", {}),
            strategy_imports=stats.get("strategy_imports", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/import/best", response_model=ImportResponse, summary="Import best matching strategy")
async def import_best_strategy(
    query: str = Query(..., description="Search query"),
    strategy_name: Optional[str] = Query(None, description="Custom strategy name"),
    preset_name: str = Query("tradingview", description="Preset name"),
    run_optimization: bool = Query(False, description="Run optimization after import"),
    background_tasks: BackgroundTasks = None,
    client=Depends(get_mcp_client),
    integrator=Depends(get_integrator)
):
    """
    Search for and import the best matching strategy
    
    Convenience endpoint that searches for scripts and imports the best match.
    Useful for quick imports based on search terms.
    """
    try:
        with client:
            import_result = client.import_best(query, strategy_name)
        
        if not import_result.get("success"):
            return ImportResponse(
                success=False,
                message="Import failed",
                error=import_result.get("error", "No matching scripts found")
            )
        
        # Get the configuration and import via integrator
        config = import_result.get("config")
        script_info = import_result.get("script_info", {})
        
        if config:
            integration_result = integrator.import_strategy(
                slug=script_info.get("slug", "unknown"),
                preset_name=preset_name
            )
            
            response = ImportResponse(
                success=integration_result.get("success", False),
                message=f"Imported best match: {script_info.get('title', 'Unknown')}",
                target_config=integration_result.get("target_config"),
                preset_name=preset_name,
                config_preview=config,
                optimization_scheduled=False
            )
            
            if run_optimization and background_tasks:
                background_tasks.add_task(
                    schedule_optimization,
                    script_info.get("slug"),
                    integration_result.get("target_config"),
                    preset_name
                )
                response.optimization_scheduled = True
                response.message += " (optimization scheduled)"
            
            return response
        else:
            return ImportResponse(
                success=False,
                message="Failed to generate configuration",
                error="Configuration generation failed"
            )
            
    except Exception as e:
        logger.error(f"Import best failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import best failed: {str(e)}")

@router.get("/validate", summary="Validate TradingView integration")
async def validate_integration():
    """
    Validate the TradingView integration system
    
    Checks all components and dependencies for proper setup.
    """
    try:
        from configdata.strategies.tradingview_integration import validate_tradingview_integration
        
        validation = validate_tradingview_integration()
        
        return {
            "validation": validation,
            "timestamp": datetime.now().isoformat(),
            "healthy": validation.get("valid", False)
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# Background task functions
async def schedule_optimization(slug: str, target_config: str, preset_name: str):
    """Background task to schedule parameter optimization"""
    try:
        logger.info(f"Scheduling optimization for {slug} in {target_config} preset {preset_name}")
        
        # This would integrate with the existing optimization pipeline
        # For now, just log the request
        
        # TODO: Integrate with worker/app/forex_scanner/optimization/
        # optimization_params = {
        #     "strategy": target_config,
        #     "preset": preset_name,
        #     "source": "tradingview_import",
        #     "slug": slug
        # }
        # run_optimization(optimization_params)
        
        logger.info(f"Optimization scheduled for {slug}")
        
    except Exception as e:
        logger.error(f"Failed to schedule optimization for {slug}: {e}")

# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Export router
__all__ = ["router"]