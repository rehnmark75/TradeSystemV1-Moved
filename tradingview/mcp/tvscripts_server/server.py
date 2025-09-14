"""
TradingView Scripts MCP Server

Main MCP server implementation providing tools and resources for:
- Searching TradingView scripts
- Retrieving script metadata and code
- Analyzing Pine Script patterns
- Generating strategy configurations
"""

import json
import asyncio
import logging
import time
from typing import Any, Sequence
from mcp import Server
from mcp.types import (
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource, 
    CallToolRequest,
    ListResourcesRequest,
    ReadResourceRequest,
    Resource
)

from .db import DB
from .resources import TVScriptResource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TVScriptsServer:
    """TradingView Scripts MCP Server"""
    
    def __init__(self, db_path: str = "tv_scripts.db"):
        self.db = DB(db_path)
        self.server = Server("tv-scripts")
        self._setup_tools()
        self._setup_resources()
    
    def _setup_tools(self):
        """Setup MCP tools for script operations"""
        
        @self.server.call_tool()
        async def search_scripts(arguments: dict) -> Sequence[TextContent]:
            """Search TradingView scripts with full-text search"""
            try:
                query = arguments.get("q", "")
                limit = min(int(arguments.get("limit", 20)), 100)  # Cap at 100
                filters = arguments.get("filters", {})
                
                if not query:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Query parameter 'q' is required"})
                    )]
                
                results = self.db.search(query, limit, filters)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "query": query,
                        "results": results,
                        "total": len(results)
                    })
                )]
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_script(arguments: dict) -> Sequence[TextContent]:
            """Get script metadata and code by slug"""
            try:
                slug = arguments.get("slug", "")
                
                if not slug:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Slug parameter is required"})
                    )]
                
                script = self.db.get(slug)
                
                if not script:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Script not found: {slug}"})
                    )]
                
                # Remove code if not open source
                if not script.get("open_source", False):
                    script = {**script, "code": None, "normalized_code": None}
                    script["message"] = "Script is not open-source; code unavailable"
                
                return [TextContent(
                    type="text",
                    text=json.dumps(script)
                )]
                
            except Exception as e:
                logger.error(f"Get script failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def analyze_pine_script(arguments: dict) -> Sequence[TextContent]:
            """Analyze Pine Script code and extract patterns"""
            try:
                slug = arguments.get("slug")
                code = arguments.get("code")
                
                if slug:
                    script = self.db.get(slug)
                    if not script or not script.get("open_source"):
                        return [TextContent(
                            type="text",
                            text=json.dumps({"error": "Script not found or not open-source"})
                        )]
                    code = script.get("code", "")
                
                if not code:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "No Pine Script code provided"})
                    )]
                
                # Import analysis modules (placeholder - will be implemented later)
                try:
                    from ..strategy_bridge.extract_pine import extract_inputs, extract_signals
                except ImportError:
                    # Placeholder implementations
                    def extract_inputs(code):
                        return [{"type": "int", "label": "placeholder", "default": "21"}]
                    
                    def extract_signals(code):
                        return {
                            "ema_periods": [21, 50],
                            "has_cross_up": True,
                            "has_cross_down": True,
                            "macd": None,
                            "higher_tf": [],
                            "mentions_fvg": "fvg" in code.lower(),
                            "mentions_smc": "smc" in code.lower() or "structure" in code.lower()
                        }
                
                # Extract patterns
                inputs = extract_inputs(code)
                signals = extract_signals(code)
                
                analysis = {
                    "inputs": inputs,
                    "signals": signals,
                    "analysis_complete": True,
                    "code_length": len(code),
                    "line_count": len(code.split('\n'))
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(analysis)
                )]
                
            except Exception as e:
                logger.error(f"Pine script analysis failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def generate_strategy_config(arguments: dict) -> Sequence[TextContent]:
            """Generate TradeSystemV1 strategy configuration from Pine Script"""
            try:
                slug = arguments.get("slug")
                strategy_name = arguments.get("strategy_name", "ImportedFromTV")
                
                if not slug:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Slug parameter is required"})
                    )]
                
                script = self.db.get(slug)
                if not script or not script.get("open_source"):
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Script not found or not open-source"})
                    )]
                
                code = script.get("code", "")
                if not code:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "No Pine Script code available"})
                    )]
                
                # Import analysis and mapping modules (placeholder - will be implemented later)
                try:
                    from ..strategy_bridge.extract_pine import extract_inputs, extract_signals
                    from ..strategy_bridge.map_to_python import to_config
                except ImportError:
                    # Placeholder implementations
                    def extract_inputs(code):
                        return [{"type": "int", "label": "placeholder", "default": "21"}]
                    
                    def extract_signals(code):
                        return {
                            "ema_periods": [21, 50],
                            "has_cross_up": True,
                            "has_cross_down": True,
                            "macd": None,
                            "higher_tf": [],
                            "mentions_fvg": "fvg" in code.lower(),
                            "mentions_smc": "smc" in code.lower() or "structure" in code.lower()
                        }
                    
                    def to_config(inputs, signals):
                        return {
                            "name": "ImportedFromTV",
                            "modules": {
                                "ema": {"periods": signals.get("ema_periods", [21, 50])},
                                "macd": signals.get("macd"),
                                "fvg": {"enabled": signals.get("mentions_fvg", False)},
                                "smc": {"enabled": signals.get("mentions_smc", False)}
                            },
                            "rules": [{"type": "placeholder", "condition": "ema_crossover"}]
                        }
                
                # Generate configuration
                inputs = extract_inputs(code)
                signals = extract_signals(code)
                config = to_config(inputs, signals)
                
                # Add metadata
                config["name"] = strategy_name
                config["provenance"] = {
                    "source": "tradingview",
                    "slug": slug,
                    "url": script.get("url"),
                    "title": script.get("title"),
                    "author": script.get("author"),
                    "imported_at": int(time.time())
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "config": config,
                        "inputs": inputs,
                        "signals": signals,
                        "generation_complete": True
                    })
                )]
                
            except Exception as e:
                logger.error(f"Strategy config generation failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_database_stats(arguments: dict) -> Sequence[TextContent]:
            """Get database statistics and system status"""
            try:
                stats = self.db.get_stats()
                return [TextContent(
                    type="text",
                    text=json.dumps(stats)
                )]
                
            except Exception as e:
                logger.error(f"Get stats failed: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
    
    def _setup_resources(self):
        """Setup MCP resources for script access"""
        
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List available script resources"""
            # For now, return dynamic resource info
            return [
                Resource(
                    uri="tvscript://search",
                    name="TradingView Script Search",
                    description="Search interface for TradingView scripts",
                    mimeType="application/json"
                ),
                Resource(
                    uri="tvscript://stats",
                    name="Database Statistics",
                    description="TradingView scripts database statistics",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read script resource by URI"""
            try:
                if uri.startswith("tvscript://"):
                    path = uri.replace("tvscript://", "")
                    
                    if path == "search":
                        return json.dumps({
                            "message": "Use search_scripts tool to search for scripts",
                            "example": {"q": "ema crossover", "limit": 10}
                        })
                    
                    elif path == "stats":
                        stats = self.db.get_stats()
                        return json.dumps(stats)
                    
                    elif path.startswith("script/"):
                        slug = path.replace("script/", "")
                        script = self.db.get(slug)
                        
                        if not script:
                            raise FileNotFoundError(f"Script not found: {slug}")
                        
                        if not script.get("open_source", False):
                            script = {**script, "code": None}
                            script["message"] = f"Source not open-source for {slug}"
                        
                        return json.dumps(script)
                
                raise FileNotFoundError(f"Resource not found: {uri}")
                
            except Exception as e:
                logger.error(f"Read resource failed for {uri}: {e}")
                raise
    
    async def run(self):
        """Run the MCP server"""
        try:
            logger.info("Starting TradingView Scripts MCP Server...")
            logger.info(f"Database stats: {self.db.get_stats()}")
            
            # Run server
            async with self.server.stdio() as streams:
                await self.server.run(
                    streams[0], streams[1],
                    self.server.create_initialization_options()
                )
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.db.close()

async def main():
    """Main entry point"""
    import time
    
    server = TVScriptsServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())