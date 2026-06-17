"""Reusable price-pattern detectors (pure numpy, no DB/pandas in the hot path)."""
from .decline_base_climb import detect_pattern, PATTERN_VERSION

__all__ = ["detect_pattern", "PATTERN_VERSION"]
