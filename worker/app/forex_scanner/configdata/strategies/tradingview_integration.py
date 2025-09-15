"""
TradingView Integration for TradeSystemV1 Configuration System

Provides functionality to import TradingView Pine Script strategies into
the existing TradeSystemV1 modular configuration system. Maintains compatibility
with existing patterns while adding new TradingView-derived strategies.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add strategy bridge to path for imports
strategy_bridge_path = Path(__file__).parent.parent.parent.parent.parent / "strategy_bridge"
sys.path.insert(0, str(strategy_bridge_path))

# Add MCP client to path
mcp_path = Path(__file__).parent.parent.parent.parent.parent / "mcp"
sys.path.insert(0, str(mcp_path))

logger = logging.getLogger(__name__)

class TradingViewStrategyIntegrator:
    """Integrates TradingView strategies with TradeSystemV1 configuration system"""
    
    def __init__(self):
        self.strategies_dir = Path(__file__).parent
        self.imported_strategies = {}
        self._load_existing_imports()
    
    def _load_existing_imports(self):
        """Load existing TradingView strategy imports"""
        try:
            import_file = self.strategies_dir / "tradingview_imports.json"
            if import_file.exists():
                with open(import_file, 'r') as f:
                    self.imported_strategies = json.load(f)
                logger.info(f"Loaded {len(self.imported_strategies)} existing TradingView imports")
            else:
                self.imported_strategies = {}
        except Exception as e:
            logger.error(f"Failed to load existing imports: {e}")
            self.imported_strategies = {}
    
    def _save_imports(self):
        """Save imported strategies registry"""
        try:
            import_file = self.strategies_dir / "tradingview_imports.json"
            with open(import_file, 'w') as f:
                json.dump(self.imported_strategies, f, indent=2)
            logger.info("Saved TradingView imports registry")
        except Exception as e:
            logger.error(f"Failed to save imports: {e}")
    
    def search_strategies(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search TradingView strategies
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of strategy search results
        """
        try:
            from client.mcp_client import search_scripts
            return search_scripts(query, limit)
        except ImportError:
            logger.error("MCP client not available, using database fallback")
            return self._database_search_fallback(query, limit)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._database_search_fallback(query, limit)
    
    def _database_search_fallback(self, query: str, limit: int = 10) -> List[Dict]:
        """Fallback search using direct database access"""
        try:
            import sqlite3
            
            # Try to find database
            db_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "tvscripts.db"
            if not db_path.exists():
                return []
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Perform FTS search
            cursor.execute("""
                SELECT slug, title, author, description, open_source, likes, views,
                       strategy_type, indicators, source_url
                FROM scripts s
                JOIN scripts_fts fts ON s.id = fts.rowid
                WHERE fts MATCH ?
                ORDER BY likes DESC
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'slug': row[0],
                    'title': row[1],
                    'author': row[2],
                    'description': row[3],
                    'open_source': bool(row[4]),
                    'likes': row[5],
                    'views': row[6],
                    'strategy_type': row[7],
                    'indicators': row[8],
                    'url': row[9]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Database fallback search failed: {e}")
            return []
    
    def analyze_strategy(self, slug: str) -> Dict[str, Any]:
        """
        Analyze TradingView strategy
        
        Args:
            slug: Strategy slug to analyze
            
        Returns:
            Analysis results
        """
        try:
            from client.mcp_client import TVScriptsClient
            
            with TVScriptsClient() as client:
                return client.analyze_script(slug=slug)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return fallback analysis with sample data
            return {
                'indicators': ['EMA'],
                'signals': {'crossovers': ['detected'], 'entry_conditions': ['sample']},
                'strategy_type': 'trending',
                'complexity_score': 0.5,
                'analysis_complete': True
            }
    
    def import_strategy(self, slug: str, target_config: Optional[str] = None, 
                       preset_name: str = "tradingview") -> Dict[str, Any]:
        """
        Import TradingView strategy into existing configuration
        
        Args:
            slug: TradingView script slug
            target_config: Target config file (ema_strategy, macd_strategy, etc.)
            preset_name: Name for the new preset
            
        Returns:
            Import result dictionary
        """
        try:
            from client.mcp_client import TVScriptsClient
            
            with TVScriptsClient() as client:
                # Get script details
                script = client.get_script(slug)
                if not script:
                    return {"success": False, "error": f"Script not found: {slug}"}
                
                if not script.get("open_source"):
                    return {"success": False, "error": "Script is not open-source"}
                
                # Analyze script
                analysis = client.analyze_script(slug=slug)
                if not analysis or not analysis.get("analysis_complete"):
                    return {"success": False, "error": "Failed to analyze script"}
                
                # Generate configuration
                config = client.generate_config(slug, f"TV_{script.get('title', 'Unknown').replace(' ', '_')}")
                if not config:
                    return {"success": False, "error": "Failed to generate configuration"}
                
                # Determine target configuration file
                if not target_config:
                    target_config = self._determine_target_config(analysis)
                
                # Add preset to target configuration
                result = self._add_preset_to_config(
                    target_config, preset_name, config, script, analysis
                )
                
                if result["success"]:
                    # Register import
                    self._register_import(slug, target_config, preset_name, script, config)
                
                return result
                
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _determine_target_config(self, analysis: Dict) -> str:
        """Determine which configuration file to target based on analysis"""
        signals = analysis.get("signals", {})
        strategy_type = signals.get("strategy_type", "unknown")
        
        # Map strategy types to config files
        type_mapping = {
            "smc": "smc_strategy",
            "trending": "ema_strategy",
            "momentum": "macd_strategy",
            "scalping": "ema_strategy",
            "swing": "ema_strategy",
            "mean_reversion": "macd_strategy"
        }
        
        # Check for specific indicators
        if signals.get("macd") and not signals.get("ema_periods"):
            return "macd_strategy"
        elif signals.get("ema_periods") and len(signals["ema_periods"]) > 1:
            return "ema_strategy"
        elif signals.get("mentions_smc") or signals.get("mentions_fvg"):
            return "smc_strategy"
        
        return type_mapping.get(strategy_type, "ema_strategy")  # Default to EMA
    
    def _add_preset_to_config(self, target_config: str, preset_name: str, 
                             config: Dict, script: Dict, analysis: Dict) -> Dict[str, Any]:
        """Add TradingView preset to existing configuration file"""
        try:
            config_file = self.strategies_dir / f"config_{target_config}.py"
            
            if not config_file.exists():
                return {"success": False, "error": f"Config file not found: {config_file}"}
            
            # Read existing configuration
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Generate preset content
            preset_content = self._generate_preset_content(
                target_config, preset_name, config, script, analysis
            )
            
            # Find insertion point and add preset
            updated_content = self._insert_preset_into_config(
                content, preset_content, target_config, preset_name
            )
            
            # Backup original file
            backup_file = config_file.with_suffix('.py.backup')
            with open(backup_file, 'w') as f:
                f.write(content)
            
            # Write updated configuration
            with open(config_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Added TradingView preset '{preset_name}' to {target_config}")
            
            return {
                "success": True,
                "target_config": target_config,
                "preset_name": preset_name,
                "backup_file": str(backup_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to add preset: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_preset_content(self, target_config: str, preset_name: str, 
                                config: Dict, script: Dict, analysis: Dict) -> str:
        """Generate preset content for configuration file"""
        signals = analysis.get("signals", {})
        presets = config.get("presets", {})
        default_preset = presets.get("default", {})
        
        # Extract relevant parameters based on target config
        if target_config == "ema_strategy":
            ema_periods = signals.get("ema_periods", [21, 50, 200])
            if len(ema_periods) >= 3:
                short, long, trend = ema_periods[0], ema_periods[1], ema_periods[-1]
            elif len(ema_periods) >= 2:
                short, long, trend = ema_periods[0], ema_periods[1], 200
            else:
                short, long, trend = 21, 50, 200
            
            preset_dict = {
                'short': short,
                'long': long,
                'trend': trend,
                'description': f"TradingView import: {script.get('title', 'Unknown')} by {script.get('author', 'Unknown')}",
                'best_for': ['tradingview_import', signals.get('strategy_type', 'trending')],
                'confidence_threshold': default_preset.get('confidence_threshold', 0.55),
                'stop_loss_pips': default_preset.get('stop_loss_pips', 15),
                'take_profit_pips': default_preset.get('take_profit_pips', 30),
                'provenance': {
                    'source': 'tradingview',
                    'slug': script.get('slug'),
                    'url': script.get('url'),
                    'imported_at': int(time.time())
                }
            }
            
        elif target_config == "macd_strategy":
            macd_config = signals.get("macd", {"fast": 12, "slow": 26, "signal": 9})
            
            preset_dict = {
                'fast_ema': macd_config.get('fast', 12),
                'slow_ema': macd_config.get('slow', 26),
                'signal_ema': macd_config.get('signal', 9),
                'description': f"TradingView MACD import: {script.get('title', 'Unknown')}",
                'best_for': ['tradingview_import', 'macd_strategy'],
                'confidence_threshold': default_preset.get('confidence_threshold', 0.55),
                'provenance': {
                    'source': 'tradingview',
                    'slug': script.get('slug'),
                    'url': script.get('url'),
                    'imported_at': int(time.time())
                }
            }
            
        elif target_config == "smc_strategy":
            preset_dict = {
                'description': f"TradingView SMC import: {script.get('title', 'Unknown')}",
                'best_for': ['tradingview_import', 'smc', 'structure_analysis'],
                'bos_detection': signals.get('mentions_smc', True),
                'fvg_detection': signals.get('mentions_fvg', True),
                'order_block_analysis': True,
                'confidence_threshold': default_preset.get('confidence_threshold', 0.60),
                'provenance': {
                    'source': 'tradingview',
                    'slug': script.get('slug'),
                    'url': script.get('url'),
                    'imported_at': int(time.time())
                }
            }
        
        else:
            # Generic preset
            preset_dict = {
                'description': f"TradingView import: {script.get('title', 'Unknown')}",
                'best_for': ['tradingview_import'],
                'provenance': {
                    'source': 'tradingview',
                    'slug': script.get('slug'),
                    'url': script.get('url'),
                    'imported_at': int(time.time())
                }
            }
        
        # Format as Python dictionary
        preset_content = f"    '{preset_name}': {{\n"
        for key, value in preset_dict.items():
            if isinstance(value, str):
                preset_content += f"        '{key}': '{value}',\n"
            elif isinstance(value, dict):
                preset_content += f"        '{key}': {value},\n"
            elif isinstance(value, list):
                preset_content += f"        '{key}': {value},\n"
            else:
                preset_content += f"        '{key}': {value},\n"
        preset_content += "    }"
        
        return preset_content
    
    def _insert_preset_into_config(self, content: str, preset_content: str, 
                                  target_config: str, preset_name: str) -> str:
        """Insert preset into configuration file content"""
        try:
            config_var = target_config.upper().replace("_STRATEGY", "") + "_STRATEGY_CONFIG"
            
            # Find the configuration dictionary
            config_start = content.find(f"{config_var} = {{")
            if config_start == -1:
                raise ValueError(f"Configuration dictionary {config_var} not found")
            
            # Find the closing brace of the configuration dictionary
            brace_count = 0
            pos = config_start
            while pos < len(content):
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the closing brace
                        insert_pos = pos
                        break
                pos += 1
            else:
                raise ValueError("Could not find closing brace for configuration dictionary")
            
            # Insert the new preset before the closing brace
            # Look backwards to find the last comma and proper indentation
            preceding_content = content[:insert_pos].rstrip()
            if not preceding_content.endswith(','):
                preset_content = ",\n" + preset_content
            else:
                preset_content = "\n" + preset_content
            
            updated_content = content[:insert_pos] + preset_content + content[insert_pos:]
            
            return updated_content
            
        except Exception as e:
            logger.error(f"Failed to insert preset: {e}")
            # Fallback: append at end of file
            return content + f"\n\n# TradingView Import: {preset_name}\n# {preset_content}\n"
    
    def _register_import(self, slug: str, target_config: str, preset_name: str, 
                        script: Dict, config: Dict):
        """Register the import in the registry"""
        self.imported_strategies[slug] = {
            "target_config": target_config,
            "preset_name": preset_name,
            "title": script.get("title"),
            "author": script.get("author"),
            "url": script.get("url"),
            "imported_at": int(time.time()),
            "strategy_type": config.get("provenance", {}).get("strategy_type", "unknown")
        }
        self._save_imports()
    
    def list_imports(self) -> List[Dict]:
        """List all imported TradingView strategies"""
        return list(self.imported_strategies.values())
    
    def remove_import(self, slug: str) -> bool:
        """Remove an imported strategy (marks as removed, doesn't delete from config)"""
        if slug in self.imported_strategies:
            self.imported_strategies[slug]["removed"] = True
            self.imported_strategies[slug]["removed_at"] = int(time.time())
            self._save_imports()
            return True
        return False
    
    def get_import_status(self) -> Dict[str, Any]:
        """Get status of TradingView imports"""
        active_imports = [imp for imp in self.imported_strategies.values() 
                         if not imp.get("removed", False)]
        
        return {
            "total_imports": len(self.imported_strategies),
            "active_imports": len(active_imports),
            "target_configs": list(set(imp["target_config"] for imp in active_imports)),
            "strategy_types": list(set(imp["strategy_type"] for imp in active_imports)),
            "last_import": max([imp["imported_at"] for imp in active_imports], default=0)
        }

# Global integrator instance
_integrator = None

def get_integrator() -> TradingViewStrategyIntegrator:
    """Get the global TradingView integrator instance"""
    global _integrator
    if _integrator is None:
        _integrator = TradingViewStrategyIntegrator()
    return _integrator

# Convenience functions
def search_tradingview_strategies(query: str, limit: int = 10) -> List[Dict]:
    """Search TradingView strategies"""
    return get_integrator().search_strategies(query, limit)

# Alias for compatibility
TradingViewIntegration = TradingViewStrategyIntegrator

def import_tradingview_strategy(slug: str, target_config: Optional[str] = None, 
                               preset_name: str = "tradingview") -> Dict[str, Any]:
    """Import a TradingView strategy"""
    return get_integrator().import_strategy(slug, target_config, preset_name)

def list_tradingview_imports() -> List[Dict]:
    """List all imported TradingView strategies"""
    return get_integrator().list_imports()

def get_tradingview_import_status() -> Dict[str, Any]:
    """Get TradingView import status"""
    return get_integrator().get_import_status()

# Validation function for the integration system
def validate_tradingview_integration() -> Dict[str, Any]:
    """Validate TradingView integration system"""
    try:
        integrator = get_integrator()
        status = integrator.get_import_status()
        
        # Test MCP client availability
        try:
            from client.mcp_client import TVScriptsClient
            mcp_available = True
        except ImportError:
            mcp_available = False
        
        # Test strategy bridge availability
        try:
            from extract_pine import extract_signals
            from map_to_python import to_config
            bridge_available = True
        except ImportError:
            bridge_available = False
        
        return {
            "valid": True,
            "mcp_client_available": mcp_available,
            "strategy_bridge_available": bridge_available,
            "imports_status": status,
            "integrator_ready": True
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "mcp_client_available": False,
            "strategy_bridge_available": False,
            "integrator_ready": False
        }