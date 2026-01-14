"""Chart styling constants for visualization."""

# Colors
COLOR_PRIMARY = "#1f77b4"
COLOR_POSITIVE = "#2ca02c"
COLOR_NEGATIVE = "#d62728"
COLOR_NEUTRAL = "#7f7f7f"
COLOR_BACKGROUND = "#f0f0f0"
COLOR_GRID = "#cccccc"

# Export COLORS for compatibility
COLORS = {
    "primary": COLOR_PRIMARY,
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral": COLOR_NEUTRAL,
    "background": COLOR_BACKGROUND,
    "grid": COLOR_GRID,
    "equity": COLOR_PRIMARY,
    "drawdown": COLOR_NEGATIVE,
    "benchmark": COLOR_NEUTRAL,
}

# Size and layout
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 8
DPI = 300
FONT_SIZE_LARGE = 12
FONT_SIZE_MEDIUM = 10
FONT_SIZE_SMALL = 8

# Grid styling
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
GRID_COLOR = COLOR_GRID

# Line styles
LINE_WIDTH = 2
LINE_WIDTH_THIN = 1

# Markers
MARKER_SIZE = 4
MARKER_SIZE_LARGE = 6
