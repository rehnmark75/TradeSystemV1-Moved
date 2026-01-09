"""
Parameter Grid Generator

Parses parameter variation specifications and generates combinations.
Supports grid syntax (start:end:step), list syntax (val1,val2,val3), and JSON.
"""

import json
import os
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridSpec:
    """Specification for a single parameter's variation"""
    name: str
    values: List[Any]
    source: str = 'unknown'  # 'range', 'list', 'json'

    def __post_init__(self):
        if not self.values:
            raise ValueError(f"GridSpec for '{self.name}' must have at least one value")


class ParameterGridGenerator:
    """
    Generate parameter combinations from various input formats.

    Supported formats:
    1. Grid syntax: "param=start:end:step" -> generates range
    2. List syntax: "param=val1,val2,val3" -> uses explicit values
    3. JSON: inline JSON string or file path
    """

    # Known parameter types for auto-conversion
    INT_PARAMS = {
        'ema_period', 'swing_lookback_bars', 'cooldown_minutes',
        'max_daily_signals', 'london_open_hour', 'ny_open_hour',
        'session_end_buffer_minutes', 'chunk_days', 'workers',
        'swing_proximity_min_distance_pips', 'swing_proximity_lookback_swings',
        'max_pullback_wait_bars', 'pullback_confirmation_bars'
    }

    FLOAT_PARAMS = {
        'fixed_stop_loss_pips', 'fixed_take_profit_pips', 'sl_buffer_pips',
        'min_confidence', 'max_confidence', 'min_risk_reward', 'max_position_size',
        'fvg_minimum_size_pips', 'displacement_atr_multiplier',
        'fib_min', 'fib_max', 'fib_pullback_min', 'fib_pullback_max',
        'high_volume_confidence', 'ema_buffer_pips', 'min_distance_from_ema_pips',
        'swing_proximity_resistance_buffer', 'swing_proximity_support_buffer',
        'pullback_offset_atr_factor', 'pullback_offset_min_pips', 'pullback_offset_max_pips',
        'min_swing_atr_multiplier', 'fallback_min_swing_pips'
    }

    BOOL_PARAMS = {
        'macd_filter_enabled', 'volume_filter_enabled', 'require_ema_alignment',
        'trend_filter_enabled', 'block_asian_session', 'weekend_filter_enabled',
        'high_impact_news_filter', 'require_liquidity_sweep', 'use_atr_stop_loss',
        'swing_proximity_enabled', 'swing_proximity_strict_mode', 'pullback_enabled',
        'use_dynamic_swing_lookback', 'use_atr_swing_validation', 'use_swing_target'
    }

    def parse_spec(self, spec: str) -> GridSpec:
        """
        Parse a parameter specification string.

        Formats:
            'param=start:end:step' -> GridSpec with range values
            'param=val1,val2,val3' -> GridSpec with explicit values

        Examples:
            'fixed_stop_loss_pips=8:12:2' -> GridSpec('fixed_stop_loss_pips', [8.0, 10.0, 12.0])
            'min_confidence=0.45,0.50,0.55' -> GridSpec('min_confidence', [0.45, 0.50, 0.55])
            'macd_filter_enabled=true,false' -> GridSpec('macd_filter_enabled', [True, False])
        """
        if '=' not in spec:
            raise ValueError(f"Invalid spec format: {spec}. Expected 'param=values'")

        name, value_part = spec.split('=', 1)
        name = name.strip()
        value_part = value_part.strip()

        if not name or not value_part:
            raise ValueError(f"Invalid spec: {spec}. Name and values are required")

        # Determine if it's range syntax or list syntax
        if ':' in value_part and ',' not in value_part:
            # Range syntax: start:end:step
            values = self._parse_range(name, value_part)
            source = 'range'
        else:
            # List syntax: val1,val2,val3
            values = self._parse_list(name, value_part)
            source = 'list'

        return GridSpec(name=name, values=values, source=source)

    def _parse_range(self, name: str, range_str: str) -> List[Any]:
        """Parse 'start:end:step' range specification"""
        parts = range_str.split(':')

        if len(parts) == 2:
            # start:end with default step
            start_str, end_str = parts
            step_str = None
        elif len(parts) == 3:
            start_str, end_str, step_str = parts
        else:
            raise ValueError(f"Invalid range format: {range_str}. Expected 'start:end' or 'start:end:step'")

        # Determine type based on parameter name or presence of decimal
        is_float = name in self.FLOAT_PARAMS or '.' in start_str or '.' in end_str

        if is_float:
            start = float(start_str)
            end = float(end_str)
            step = float(step_str) if step_str else 1.0

            # Generate float range
            values = []
            current = start
            while current <= end + 1e-9:  # Small epsilon for float comparison
                values.append(round(current, 4))
                current += step
        else:
            start = int(start_str)
            end = int(end_str)
            step = int(step_str) if step_str else 1

            values = list(range(start, end + 1, step))

        if not values:
            raise ValueError(f"Range {range_str} generated no values")

        return values

    def _parse_list(self, name: str, list_str: str) -> List[Any]:
        """Parse 'val1,val2,val3' list specification"""
        raw_values = [v.strip() for v in list_str.split(',')]
        values = []

        for raw in raw_values:
            if not raw:
                continue
            values.append(self._convert_value(name, raw))

        if not values:
            raise ValueError(f"List '{list_str}' generated no values")

        return values

    def _convert_value(self, name: str, value: str) -> Any:
        """Convert string value to appropriate type based on parameter name"""
        value_lower = value.lower()

        # Boolean conversion
        if name in self.BOOL_PARAMS or value_lower in ('true', 'false'):
            return value_lower == 'true'

        # Integer conversion
        if name in self.INT_PARAMS:
            return int(value)

        # Float conversion
        if name in self.FLOAT_PARAMS or '.' in value:
            return float(value)

        # Try numeric conversion
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value  # Keep as string

    def from_json(self, json_input: str) -> List[GridSpec]:
        """
        Parse JSON parameter grid.

        Args:
            json_input: Either inline JSON string or path to JSON file

        Expected JSON format:
            {
                "fixed_stop_loss_pips": [8, 10, 12],
                "min_confidence": [0.45, 0.50, 0.55]
            }
        """
        # Check if it's a file path
        if os.path.isfile(json_input):
            with open(json_input, 'r') as f:
                data = json.load(f)
        else:
            # Parse as inline JSON
            try:
                data = json.loads(json_input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")

        if not isinstance(data, dict):
            raise ValueError("JSON must be an object with parameter names as keys")

        specs = []
        for name, values in data.items():
            if not isinstance(values, list):
                values = [values]

            # Convert values to appropriate types
            converted = [self._convert_value(name, str(v)) for v in values]
            specs.append(GridSpec(name=name, values=converted, source='json'))

        return specs

    def generate_combinations(
        self,
        specs: List[GridSpec],
        max_combinations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid specs.

        Uses Cartesian product to generate all possible combinations.

        Args:
            specs: List of GridSpec objects
            max_combinations: Optional limit on number of combinations

        Returns:
            List of parameter dictionaries
        """
        if not specs:
            return []

        # Extract names and values
        names = [spec.name for spec in specs]
        value_lists = [spec.values for spec in specs]

        # Calculate total combinations
        total = 1
        for values in value_lists:
            total *= len(values)

        if max_combinations and total > max_combinations:
            logger.warning(
                f"Parameter grid generates {total} combinations, "
                f"exceeding limit of {max_combinations}"
            )

        # Generate combinations
        combinations = []
        for combo in product(*value_lists):
            param_dict = dict(zip(names, combo))
            combinations.append(param_dict)

            if max_combinations and len(combinations) >= max_combinations:
                logger.warning(f"Truncated to {max_combinations} combinations")
                break

        return combinations

    def parse_all(
        self,
        vary_specs: Optional[List[str]] = None,
        json_input: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse all parameter specifications and generate combinations.

        Args:
            vary_specs: List of 'param=spec' strings from --vary arguments
            json_input: JSON string or file path from --vary-json argument

        Returns:
            List of parameter combination dictionaries
        """
        all_specs: List[GridSpec] = []

        # Parse --vary arguments
        if vary_specs:
            for spec_str in vary_specs:
                try:
                    spec = self.parse_spec(spec_str)
                    all_specs.append(spec)
                    logger.info(f"Parsed variation: {spec.name} = {spec.values}")
                except ValueError as e:
                    logger.error(f"Failed to parse spec '{spec_str}': {e}")
                    raise

        # Parse --vary-json argument
        if json_input:
            try:
                json_specs = self.from_json(json_input)
                all_specs.extend(json_specs)
                for spec in json_specs:
                    logger.info(f"Parsed JSON variation: {spec.name} = {spec.values}")
            except ValueError as e:
                logger.error(f"Failed to parse JSON: {e}")
                raise

        # Generate combinations
        combinations = self.generate_combinations(all_specs)
        logger.info(f"Generated {len(combinations)} parameter combinations")

        return combinations

    def summary(self, specs: List[GridSpec]) -> str:
        """Generate a summary of the parameter grid"""
        if not specs:
            return "No parameters to vary"

        lines = ["Parameter Grid:"]
        total = 1

        for spec in specs:
            lines.append(f"  {spec.name}: {spec.values} ({len(spec.values)} values)")
            total *= len(spec.values)

        lines.append(f"Total combinations: {total}")
        return "\n".join(lines)
