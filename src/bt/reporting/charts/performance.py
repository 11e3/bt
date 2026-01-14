"""Performance analysis charts.

Creates charts for return distributions, risk metrics,
and trade analysis.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..charts.base import BaseChart, ChartFormatter, ChartStyle


class ReturnDistributionChart(BaseChart):
    """Chart for visualizing return distributions."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate return distribution chart.

        Args:
            data: Dictionary with 'returns' array and optional yearly returns

        Returns:
            Matplotlib figure with return distribution
        """
        returns = data.get("returns", [])
        if not returns:
            raise ValueError("No returns data provided")

        returns_array = np.array(returns)

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.config.get("figure_size", (14, 10))
        )

        # Histogram of returns
        ax1.hist(
            returns_array,
            bins=50,
            color=ChartStyle.COLORS["primary"],
            alpha=0.7,
            edgecolor=ChartStyle.COLORS["dark"],
        )
        ax1.set_title("Return Distribution")
        ax1.set_xlabel("Daily Return")
        ax1.set_ylabel("Frequency")

        # QQ plot
        from scipy import stats

        stats.probplot(returns_array, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot vs Normal Distribution")

        # Box plot
        ax3.boxplot(
            returns_array,
            patch_artist=True,
            boxprops={"facecolor": ChartStyle.COLORS["primary"]},
            medianprops={"color": ChartStyle.COLORS["secondary"]},
        )
        ax3.set_title("Return Box Plot")
        ax3.set_ylabel("Return")
        ax3.set_xticklabels([])

        # Cumulative returns
        cumulative_returns = np.cumprod(1 + returns_array)
        ax4.plot(cumulative_returns, color=ChartStyle.COLORS["primary"])
        ax4.set_title("Cumulative Returns")
        ax4.set_ylabel("Cumulative Return")
        ax4.set_xlabel("Trading Days")
        ax4.axhline(y=1, color=ChartStyle.COLORS["neutral"], linestyle="--", alpha=0.7)

        # Style all axes
        for ax in [ax1, ax2, ax3, ax4]:
            self.apply_style(ax)

        plt.tight_layout()
        return fig


class RiskMetricsChart(BaseChart):
    """Chart for visualizing risk metrics."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate risk metrics visualization.

        Args:
            data: Dictionary with performance metrics

        Returns:
            Matplotlib figure with risk analysis
        """
        # Extract risk metrics
        sharpe = data.get("sharpe", 0)
        sortino = data.get("sortino", 0)
        max_dd = data.get("max_drawdown", 0)
        calmar = data.get("calmar", 0)
        var_95 = data.get("var_95", 0)

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.config.get("figure_size", (14, 10))
        )

        # Risk metrics bar chart
        metrics = ["Sharpe", "Sortino", "Calmar"]
        values = [sharpe, sortino, calmar]
        colors = [
            ChartStyle.COLORS["success"],
            ChartStyle.COLORS["info"],
            ChartStyle.COLORS["primary"],
        ]

        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title("Risk-Adjusted Performance Metrics")
        ax1.set_ylabel("Ratio")
        ax1.axhline(
            y=1.0,
            color=ChartStyle.COLORS["neutral"],
            linestyle="--",
            alpha=0.7,
            label="Risk-Free Rate",
        )

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Drawdown visualization
        equity_curve = data.get("equity_curve", [])
        if equity_curve:
            drawdown_values = self._calculate_drawdown(equity_curve)

            ax2.fill_between(
                range(len(drawdown_values)),
                drawdown_values,
                0,
                color=ChartStyle.COLORS["danger"],
                alpha=0.3,
            )
            ax2.plot(drawdown_values, color=ChartStyle.COLORS["secondary"])
            ax2.set_title(f"Drawdown Profile (Max: {ChartFormatter.format_percentage(max_dd)})")
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Trading Days")

        # Risk table
        ax3.axis("off")
        risk_data = [
            ["Metric", "Value"],
            ["Max Drawdown", f"{ChartFormatter.format_percentage(max_dd)}"],
            ["95% VaR", f"{ChartFormatter.format_percentage(var_95)}"],
            ["Calmar Ratio", f"{calmar:.2f}"],
        ]

        table = ax3.table(
            cellText=risk_data[1:],
            colLabels=risk_data[0],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax3.set_title("Risk Summary")

        # Pie chart of risk contribution (placeholder)
        ax4.pie(
            [abs(sharpe), abs(sortino), abs(calmar)],
            labels=["Sharpe", "Sortino", "Calmar"],
            colors=[
                ChartStyle.COLORS["primary"],
                ChartStyle.COLORS["info"],
                ChartStyle.COLORS["secondary"],
            ],
        )
        ax4.set_title("Risk Metric Comparison")

        # Style all axes
        for ax in [ax1, ax2, ax3, ax4]:
            self.apply_style(ax)

        plt.tight_layout()
        return fig

    def _calculate_drawdown(self, equity_curve: list[float]) -> list[float]:
        """Calculate drawdown from equity curve."""
        if len(equity_curve) < 2:
            return []

        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak

        return drawdown.tolist()


class TradeAnalysisChart(BaseChart):
    """Chart for trade analysis and statistics."""

    def generate(self, data: dict[str, Any]) -> plt.Figure:
        """Generate trade analysis charts.

        Args:
            data: Dictionary with trade information

        Returns:
            Matplotlib figure with trade analysis
        """
        trades = data.get("trades", [])
        if not trades:
            raise ValueError("No trade data provided")

        # Extract trade metrics
        pnl_values = [float(trade.get("pnl", 0)) for trade in trades]
        returns_pct = [float(trade.get("return_pct", 0)) for trade in trades]

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.config.get("figure_size", (14, 10))
        )

        # PnL distribution
        winning_trades = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades = [pnl for pnl in pnl_values if pnl < 0]

        ax1.hist(
            [winning_trades, losing_trades],
            bins=30,
            color=[ChartStyle.COLORS["success"], ChartStyle.COLORS["danger"]],
            alpha=0.7,
            label=["Winning", "Losing"],
        )
        ax1.set_title("P&L Distribution")
        ax1.set_xlabel("P&L")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Return distribution
        ax2.hist(returns_pct, bins=30, color=ChartStyle.COLORS["primary"], alpha=0.7)
        ax2.axvline(
            x=0, color=ChartStyle.COLORS["neutral"], linestyle="--", alpha=0.7, label="Break Even"
        )
        ax2.set_title("Return Distribution (%)")
        ax2.set_xlabel("Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        # Trade statistics
        win_rate = len(winning_trades) / len(pnl_values) if pnl_values else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else 0

        ax3.axis("off")
        stats_data = [
            ["Metric", "Value"],
            ["Total Trades", len(trades)],
            ["Win Rate", f"{ChartFormatter.format_percentage(win_rate)}"],
            ["Avg Win", f"{ChartFormatter.format_currency(avg_win)}"],
            ["Avg Loss", f"{ChartFormatter.format_currency(avg_loss)}"],
            ["Profit Factor", f"{profit_factor:.2f}"],
        ]

        table = ax3.table(
            cellText=stats_data[1:],
            colLabels=stats_data[0],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax3.set_title("Trade Statistics")

        # Trade timeline
        if trades:
            trade_times = list(range(len(trades)))
            cumulative_pnl = np.cumsum(pnl_values)

            ax4.plot(trade_times, cumulative_pnl, color=ChartStyle.COLORS["primary"])
            ax4.fill_between(
                trade_times,
                cumulative_pnl,
                0,
                where=np.array(cumulative_pnl) >= 0,
                color=ChartStyle.COLORS["success"],
                alpha=0.3,
                label="Profit",
            )
            ax4.fill_between(
                trade_times,
                cumulative_pnl,
                0,
                where=np.array(cumulative_pnl) < 0,
                color=ChartStyle.COLORS["danger"],
                alpha=0.3,
                label="Loss",
            )
            ax4.set_title("Cumulative P&L Timeline")
            ax4.set_xlabel("Trade Number")
            ax4.set_ylabel("Cumulative P&L")
            ax4.axhline(y=0, color=ChartStyle.COLORS["neutral"], linestyle="--", alpha=0.7)
            ax4.legend()

        # Style all axes
        for ax in [ax1, ax2, ax3, ax4]:
            self.apply_style(ax)

        plt.tight_layout()
        return fig
