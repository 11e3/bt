"""Equity curve chart generation.

Specialized module for creating equity curve visualizations
with drawdown overlays and performance annotations.
"""

from datetime import datetime
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .base import BaseChart, ChartDataValidator, ChartFormatter, ChartStyle


class EquityCurveChart(BaseChart):
    """Chart generator for equity curves with drawdown overlays."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate equity curve chart.

        Args:
            data: Dictionary with 'dates' and 'values' keys

        Returns:
            Matplotlib figure with equity curve and drawdown
        """
        # Validate data
        validation_errors = ChartDataValidator.validate_performance_data(data)
        if validation_errors:
            raise ValueError(f"Invalid data: {', '.join(validation_errors)}")

        dates = data["dates"]
        values = [float(v) for v in data["values"]]
        initial_value = values[0]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=self.config.get("figure_size", (14, 8)),
            gridspec_kw={"height_ratios": [3, 1]},
            dpi=self.config.get("dpi", 100),
        )

        # Plot equity curve
        ax1.plot(
            dates, values, color=ChartStyle.COLORS["primary"], linewidth=2, label="Portfolio Value"
        )

        # Add initial value line
        ax1.axhline(
            y=initial_value,
            color=ChartStyle.COLORS["neutral"],
            linestyle="--",
            alpha=0.7,
            label=f"Initial: {ChartFormatter.format_currency(initial_value)}",
        )

        # Calculate and plot drawdown
        drawdown_values = self._calculate_drawdown(values)
        ax2.fill_between(
            dates,
            drawdown_values,
            0,
            color=ChartStyle.COLORS["danger"],
            alpha=0.3,
            label="Drawdown",
        )

        # Style both axes
        self.apply_style(ax1)
        self.apply_style(ax2)

        ax1.set_ylabel("Portfolio Value")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(bottom=max(min(drawdown_values) * 1.1, -100))

        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 12)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def _calculate_drawdown(self, values: list[float]) -> list[float]:
        """Calculate drawdown percentages."""
        peak = values[0]
        drawdowns = []

        for value in values:
            if value > peak:
                peak = value
            dd = ((value - peak) / peak) * 100 if peak > 0 else 0
            drawdowns.append(dd)

        return drawdowns


class LogEquityChart(BaseChart):
    """Logarithmic equity curve chart for better visualization of returns."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate logarithmic equity curve."""
        dates = data["dates"]
        values = [float(v) for v in data["values"]]

        fig, ax = plt.subplots(figsize=self.config.get("figure_size", (12, 6)))

        # Plot log equity curve
        ax.semilogy(dates, values, color=ChartStyle.COLORS["primary"], linewidth=2)

        self.apply_style(ax)
        ax.set_ylabel("Portfolio Value (log scale)")
        ax.set_title("Logarithmic Equity Curve")

        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        plt.tight_layout()
        return fig


class PerformanceAnnotationChart(BaseChart):
    """Chart with performance annotations and statistics."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate equity chart with performance annotations."""
        dates = data["dates"]
        values = [float(v) for v in data["values"]]

        fig, ax = plt.subplots(figsize=self.config.get("figure_size", (14, 8)))

        # Plot equity curve
        ax.plot(dates, values, color=ChartStyle.COLORS["primary"], linewidth=2)

        # Add performance annotations
        self._add_performance_annotations(ax, dates, values, data.get("performance_data", {}))

        self.apply_style(ax)
        ax.set_ylabel("Portfolio Value")
        ax.set_title("Equity Curve with Performance Metrics")

        plt.tight_layout()
        return fig

    def _add_performance_annotations(
        self, ax, _dates: list[datetime], values: list[float], perf_data: dict[str, Any]
    ) -> None:
        """Add performance statistics as text annotations."""
        if not perf_data:
            return

        # Calculate key statistics
        total_return = ((values[-1] / values[0]) - 1) * 100
        max_dd = min(self._calculate_drawdown(values))

        # Create annotation text
        stats_text = (
            f"Total Return: {ChartFormatter.format_percentage(total_return)}\n"
            f"Max Drawdown: {ChartFormatter.format_percentage(max_dd)}\n"
            f"Sharpe Ratio: {perf_data.get('sharpe', 'N/A')}\n"
            f"Win Rate: {ChartFormatter.format_percentage(perf_data.get('win_rate', 0))}"
        )

        # Add annotation box
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            family="monospace",
        )
