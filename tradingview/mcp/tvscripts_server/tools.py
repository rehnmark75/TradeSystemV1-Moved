"""
MCP Tools for TradingView Scripts Server

Defines the tool schemas and implementations for the MCP server.
Each tool provides specific functionality for script search, analysis, and configuration generation.
"""

from mcp.types import Tool

# Tool schema definitions
SEARCH_SCRIPTS_TOOL = Tool(
    name="search_scripts",
    description="Full-text search TradingView open-source scripts with advanced filtering",
    inputSchema={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": "Search query (e.g., 'ema crossover', 'macd strategy', 'smart money concepts')"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of results (default: 20, max: 100)",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            },
            "filters": {
                "type": "object",
                "description": "Optional search filters",
                "properties": {
                    "open_source_only": {
                        "type": "boolean",
                        "description": "Only return open-source scripts (default: true)",
                        "default": True
                    },
                    "script_type": {
                        "type": "string",
                        "description": "Filter by script type (indicator, strategy, library, etc.)"
                    },
                    "min_likes": {
                        "type": "number",
                        "description": "Minimum number of likes",
                        "minimum": 0
                    },
                    "author": {
                        "type": "string",
                        "description": "Filter by specific author"
                    }
                }
            }
        },
        "required": ["q"]
    }
)

GET_SCRIPT_TOOL = Tool(
    name="get_script",
    description="Get detailed script information including metadata and Pine Script code (if open-source)",
    inputSchema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Script slug identifier (from search results)"
            }
        },
        "required": ["slug"]
    }
)

ANALYZE_PINE_SCRIPT_TOOL = Tool(
    name="analyze_pine_script",
    description="Analyze Pine Script code to extract indicators, signals, and patterns",
    inputSchema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Script slug to analyze (will retrieve code automatically)"
            },
            "code": {
                "type": "string",
                "description": "Pine Script code to analyze directly"
            }
        },
        "oneOf": [
            {"required": ["slug"]},
            {"required": ["code"]}
        ]
    }
)

GENERATE_STRATEGY_CONFIG_TOOL = Tool(
    name="generate_strategy_config",
    description="Generate TradeSystemV1 strategy configuration from Pine Script analysis",
    inputSchema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Script slug to generate configuration for"
            },
            "strategy_name": {
                "type": "string",
                "description": "Name for the generated strategy configuration",
                "default": "ImportedFromTV"
            },
            "preset_type": {
                "type": "string",
                "description": "Strategy preset type (aggressive, conservative, scalping, etc.)",
                "enum": ["aggressive", "conservative", "scalping", "swing", "default"],
                "default": "default"
            }
        },
        "required": ["slug"]
    }
)

GET_DATABASE_STATS_TOOL = Tool(
    name="get_database_stats",
    description="Get database statistics including script counts, types, and system status",
    inputSchema={
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "description": "Include detailed breakdown by categories",
                "default": False
            }
        }
    }
)

CLASSIFY_STRATEGY_TOOL = Tool(
    name="classify_strategy",
    description="Classify strategy based on Pine Script patterns and indicators",
    inputSchema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Script slug to classify"
            },
            "code": {
                "type": "string",
                "description": "Pine Script code to classify directly"
            }
        },
        "oneOf": [
            {"required": ["slug"]},
            {"required": ["code"]}
        ]
    }
)

IMPORT_STRATEGY_TOOL = Tool(
    name="import_strategy",
    description="Import TradingView strategy into TradeSystemV1 configuration system",
    inputSchema={
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": "Script slug to import"
            },
            "target_config": {
                "type": "string",
                "description": "Target configuration file (e.g., 'ema_strategy', 'macd_strategy')"
            },
            "preset_name": {
                "type": "string",
                "description": "Name for the new preset",
                "default": "tradingview_import"
            },
            "run_optimization": {
                "type": "boolean",
                "description": "Run parameter optimization after import",
                "default": True
            }
        },
        "required": ["slug"]
    }
)

# All available tools
ALL_TOOLS = [
    SEARCH_SCRIPTS_TOOL,
    GET_SCRIPT_TOOL,
    ANALYZE_PINE_SCRIPT_TOOL,
    GENERATE_STRATEGY_CONFIG_TOOL,
    GET_DATABASE_STATS_TOOL,
    CLASSIFY_STRATEGY_TOOL,
    IMPORT_STRATEGY_TOOL
]

# Tool categories for organization
SEARCH_TOOLS = [SEARCH_SCRIPTS_TOOL, GET_SCRIPT_TOOL, GET_DATABASE_STATS_TOOL]
ANALYSIS_TOOLS = [ANALYZE_PINE_SCRIPT_TOOL, CLASSIFY_STRATEGY_TOOL]
GENERATION_TOOLS = [GENERATE_STRATEGY_CONFIG_TOOL, IMPORT_STRATEGY_TOOL]