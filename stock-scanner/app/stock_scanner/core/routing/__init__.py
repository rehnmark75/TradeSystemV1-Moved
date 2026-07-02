"""Per-cell edge-map router (foundation layer).

Character-cell tagging + learned (scanner x cell) edge-map. Data/analysis only;
nothing here changes any execution path on its own.
"""

from stock_scanner.core.routing.cell_tagger import (
    classify_cell,
    tag_signal_row,
    ADX_TREND_MIN,
    ADX_RANGE_MAX,
    ATR_LOW_MAX,
    ATR_HIGH_MIN,
    RVOL_THIN_MAX,
    RVOL_HIGH_MIN,
)

__all__ = [
    "classify_cell",
    "tag_signal_row",
    "ADX_TREND_MIN",
    "ADX_RANGE_MAX",
    "ATR_LOW_MAX",
    "ATR_HIGH_MIN",
    "RVOL_THIN_MAX",
    "RVOL_HIGH_MIN",
]
