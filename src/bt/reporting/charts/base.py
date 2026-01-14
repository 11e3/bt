"""Base chart functionality for all chart types.

Provides foundation classes and common functionality
for chart generation across the reporting system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any

import matplotlib.pyplot as plt


class BaseChart(ABC):
    """Base class for all chart generators."""

    def __init__(self, **config):
        self.config = config
        self.style_config = self._get_default_style()

    def _get_default_style(self) -> dict[str, Any]:
        """Get default styling configuration."""
        return {
            "figure_size": (12, 8),
            "dpi": 100,
            "font_family": "DejaVu Sans",
            "grid": True,
            "grid_alpha": 0.3,
            "title_fontsize": 14,
            "label_fontsize": 10,
            "legend_fontsize": 10,
        }

    def apply_style(self, ax) -> None:
        """Apply consistent styling to chart axis."""
        style = self.style_config

        if style.get("grid", True):
            ax.grid(True, alpha=style.get("grid_alpha", 0.3))

        ax.set_xlabel(ax.get_xlabel(), fontsize=style.get("label_fontsize", 10))
        ax.set_ylabel(ax.get_ylabel(), fontsize=style.get("label_fontsize", 10))
        ax.set_title(ax.get_title(), fontsize=style.get("title_fontsize", 14), pad=20)

        if hasattr(ax, "legend") and ax.get_legend():
            ax.legend(fontsize=style.get("legend_fontsize", 10))

    @abstractmethod
    def generate(self, data: Any) -> plt.Figure:
        """Generate the chart.

        Args:
            data: Data to visualize

        Returns:
            Matplotlib figure
        """
        pass

    def save(self, figure: plt.Figure, filepath: str, **kwargs) -> None:
        """Save chart to file with consistent settings.

        Args:
            figure: Matplotlib figure
            filepath: Output file path
            **kwargs: Additional save arguments
        """
        default_kwargs = {
            "dpi": self.style_config.get("dpi", 100),
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
        }
        default_kwargs.update(kwargs)

        figure.savefig(filepath, **default_kwargs)

    def close(self, figure: plt.Figure) -> None:
        """Clean up figure resources."""
        plt.close(figure)


class ChartStyle:
    """Chart styling configuration."""

    # Color palettes
    COLORS = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#28A745",
        "danger": "#DC3545",
        "warning": "#FFC107",
        "info": "#17A2B8",
        "light": "#F8F9FA",
        "dark": "#343A40",
    }

    PALETTES = {
        "default": ["#2E86AB", "#A23B72", "#28A745", "#DC3545", "#FFC107"],
        "profit": ["#28A745", "#20C997", "#28A745", "#20C997"],
        "loss": ["#DC3545", "#E74C3C", "#DC3545", "#E74C3C"],
        "neutral": ["#6C757D", "#95A5A6", "#6C757D", "#95A5A6"],
    }


class ChartDataValidator:
    """Validates data before chart generation."""

    @staticmethod
    def validate_timeseries(dates: list[datetime], values: list[Decimal]) -> bool:
        """Validate time series data."""
        if not dates or not values:
            return False

        if len(dates) != len(values):
            return False

        if len(dates) < 2:
            return False

        # Check for chronological order
        sorted_dates = sorted(dates)
        return dates == sorted_dates

    @staticmethod
    def validate_performance_data(data: dict[str, Any]) -> list[str]:
        """Validate performance metrics data."""
        errors = []

        required_keys = ["total_return", "cagr", "mdd", "sharpe", "sortino"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required metric: {key}")

        if "equity_curve" in data:
            equity_data = data["equity_curve"]
            if not ChartDataValidator.validate_timeseries(
                equity_data.get("dates", []), equity_data.get("values", [])
            ):
                errors.append("Invalid equity curve data")

        return errors


class ChartFormatter:
    """Formats numbers and dates for chart display."""

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format value as percentage."""
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_currency(value: float, currency: str = "", decimals: int = 2) -> str:
        """Format value as currency."""
        if currency:
            return f"{currency}{value:,.{decimals}f}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def format_large_number(value: float) -> str:
        """Format large numbers with suffixes."""
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:.1f}"
