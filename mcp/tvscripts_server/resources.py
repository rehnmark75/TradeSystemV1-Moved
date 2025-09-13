"""
MCP Resources for TradingView Scripts

Defines resource handlers for accessing TradingView scripts
via MCP resource protocol.
"""

import json
import logging
from typing import List, Optional
from mcp.types import Resource

logger = logging.getLogger(__name__)

class TVScriptResource(Resource):
    """Resource handler for TradingView scripts"""
    
    def __init__(self, db):
        self.db = db
        super().__init__()
    
    @property
    def name(self) -> str:
        return "tvscript"
    
    async def list_resources(self) -> List[Resource]:
        """List available script resources"""
        try:
            # Get recent/popular scripts
            recent_scripts = self.db.search("", limit=10, filters={'open_source_only': True})
            
            resources = [
                Resource(
                    uri="tvscript://search",
                    name="Script Search Interface",
                    description="Full-text search interface for TradingView scripts",
                    mimeType="application/json"
                ),
                Resource(
                    uri="tvscript://stats",
                    name="Database Statistics",
                    description="TradingView scripts database statistics and summary",
                    mimeType="application/json"
                )
            ]
            
            # Add individual script resources
            for script in recent_scripts:
                resources.append(Resource(
                    uri=f"tvscript://script/{script['slug']}",
                    name=f"{script['title']} by {script['author']}",
                    description=f"TradingView script: {script.get('description', script['title'])}",
                    mimeType="text/x-pinescript" if script.get('open_source') else "application/json"
                ))
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            return []
    
    async def read_resource(self, uri: str) -> str:
        """Read resource content by URI"""
        try:
            if not uri.startswith("tvscript://"):
                raise ValueError(f"Invalid URI scheme: {uri}")
            
            path = uri.replace("tvscript://", "")
            
            if path == "search":
                return json.dumps({
                    "interface": "TradingView Script Search",
                    "description": "Use the search_scripts tool to find scripts",
                    "example_queries": [
                        "ema crossover",
                        "macd strategy",
                        "smart money concepts",
                        "fair value gap",
                        "bollinger bands"
                    ],
                    "filters": {
                        "open_source_only": "bool - only return open-source scripts",
                        "script_type": "str - filter by script type",
                        "min_likes": "int - minimum likes threshold"
                    }
                })
            
            elif path == "stats":
                stats = self.db.get_stats()
                return json.dumps({
                    "database_statistics": stats,
                    "description": "TradingView scripts database overview",
                    "last_updated": "dynamic"
                })
            
            elif path.startswith("script/"):
                slug = path.replace("script/", "")
                script = self.db.get(slug)
                
                if not script:
                    raise FileNotFoundError(f"Script not found: {slug}")
                
                if script.get('open_source', False):
                    # Return Pine Script code
                    code = script.get('code', '')
                    if not code:
                        return f"// Script {slug} found but no code available"
                    return code
                else:
                    # Return metadata only
                    return json.dumps({
                        "slug": script['slug'],
                        "title": script['title'],
                        "author": script['author'],
                        "description": script.get('description', ''),
                        "open_source": False,
                        "message": "Script is not open-source; code unavailable",
                        "url": script.get('url', '')
                    })
            
            else:
                raise FileNotFoundError(f"Resource not found: {path}")
                
        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            raise
    
    def get_resource_info(self, slug: str) -> Optional[dict]:
        """Get resource information for a script"""
        try:
            script = self.db.get(slug)
            if not script:
                return None
            
            return {
                "uri": f"tvscript://script/{slug}",
                "name": f"{script['title']} by {script['author']}",
                "description": script.get('description', script['title']),
                "mimeType": "text/x-pinescript" if script.get('open_source') else "application/json",
                "open_source": script.get('open_source', False),
                "metadata": {
                    "likes": script.get('likes_count', 0),
                    "uses": script.get('uses_count', 0),
                    "tags": script.get('tags', ''),
                    "url": script.get('url', '')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource info for {slug}: {e}")
            return None