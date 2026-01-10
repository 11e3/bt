"""Performance visualization module.

Generates charts for backtest results including:
- Equity curve with drawdown
- Yearly returns bar chart
- WFA window performance
- Market regime analysis
"""

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bt.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from bt.domain.models import PerformanceMetrics

logger = get_logger(__name__)

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "equity": "#2E86AB",
    "drawdown": "#E94F37",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#95A5A6",
    "benchmark": "#7F8C8D",
}


def plot_equity_curve(
    equity_curve: list[Decimal],
    dates: list[datetime],
    title: str = "Equity Curve",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot equity curve with drawdown overlay.

    Args:
        equity_curve: List of portfolio values
        dates: Corresponding dates
        title: Chart title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    equity = np.array([float(e) for e in equity_curve])

    # Align lengths (take minimum)
    min_len = min(len(equity), len(dates))
    equity = equity[:min_len]
    dates = dates[:min_len]

    # Calculate drawdown
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True
    )


    # Equity curve (log scale)
    ax1.semilogy(dates, equity, color=COLORS["equity"], linewidth=1.5, label="Equity")
    ax1.fill_between(dates, equity, alpha=0.3, color=COLORS["equity"])
    ax1.set_ylabel("Equity (KRW, log scale)", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Drawdown
    ax2.fill_between(dates, drawdown, 0, color=COLORS["drawdown"], alpha=0.7)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(min(drawdown) * 1.1, 5)
    ax2.grid(True, alpha=0.3)

    # Add MDD annotation
    mdd_idx = np.argmin(drawdown)
    mdd_value = drawdown[mdd_idx]
    ax2.annotate(
        f"MDD: {mdd_value:.1f}%",
        xy=(dates[mdd_idx], mdd_value),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color=COLORS["drawdown"],
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Equity curve saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_yearly_returns(
    yearly_returns: dict[int, Decimal],
    title: str = "Yearly Returns",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot yearly returns as bar chart.

    Args:
        yearly_returns: Dictionary of year -> return percentage
        title: Chart title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    years = sorted(yearly_returns.keys())
    returns = [float(yearly_returns[y]) for y in years]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLORS["positive"] if r >= 0 else COLORS["negative"] for r in returns]
    bars = ax.bar(years, returns, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, ret in zip(bars, returns, strict=False):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax.annotate(
            f"{ret:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(years)
    ax.grid(True, axis="y", alpha=0.3)

    # Add average line
    avg_return = np.mean(returns)
    ax.axhline(
        y=avg_return,
        color=COLORS["benchmark"],
        linestyle="--",
        linewidth=1.5,
        label=f"Avg: {avg_return:.1f}%",
    )
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Yearly returns chart saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_wfa_results(
    window_results: list[dict],
    title: str = "Walk Forward Analysis Results",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot WFA window-by-window performance.

    Args:
        window_results: List of dictionaries with 'cagr', 'mdd', 'win_rate' keys
                       or nested structure with 'results' key containing metrics
        title: Chart title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    n_windows = len(window_results)
    windows = list(range(1, n_windows + 1))

    # Handle both flat and nested result structures
    def get_metric(r: dict, key: str) -> float:
        if "results" in r:
            return r["results"].get(key, 0)
        return r.get(key, 0)

    cagrs = [get_metric(r, "cagr") for r in window_results]
    mdds = [get_metric(r, "mdd") for r in window_results]
    win_rates = [get_metric(r, "win_rate") for r in window_results]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # CAGR
    colors = [COLORS["positive"] if c >= 0 else COLORS["negative"] for c in cagrs]
    axes[0].bar(windows, cagrs, color=colors, edgecolor="white", alpha=0.8)
    axes[0].axhline(y=0, color="black", linewidth=0.8)
    axes[0].axhline(
        y=np.median(cagrs),
        color=COLORS["benchmark"],
        linestyle="--",
        label=f"Median: {np.median(cagrs):.1f}%",
    )
    axes[0].set_ylabel("CAGR (%)", fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.3)

    # MDD
    axes[1].bar(windows, mdds, color=COLORS["drawdown"], edgecolor="white", alpha=0.8)
    axes[1].axhline(
        y=np.mean(mdds),
        color=COLORS["benchmark"],
        linestyle="--",
        label=f"Avg: {np.mean(mdds):.1f}%",
    )
    axes[1].set_ylabel("MDD (%)", fontsize=11)
    axes[1].legend(loc="lower right")
    axes[1].grid(True, axis="y", alpha=0.3)

    # Win Rate
    axes[2].bar(windows, win_rates, color=COLORS["equity"], edgecolor="white", alpha=0.8)
    axes[2].axhline(
        y=np.mean(win_rates),
        color=COLORS["benchmark"],
        linestyle="--",
        label=f"Avg: {np.mean(win_rates):.1f}%",
    )
    axes[2].set_xlabel("Window", fontsize=11)
    axes[2].set_ylabel("Win Rate (%)", fontsize=11)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"WFA results chart saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_market_regime_analysis(
    equity_curve: list[Decimal],
    dates: list[datetime],
    regime_labels: list[str] | None = None,
    title: str = "Market Regime Analysis",
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot equity curve with market regime highlighting.

    If regime_labels not provided, auto-detect based on returns.

    Args:
        equity_curve: List of portfolio values
        dates: Corresponding dates
        regime_labels: Optional list of regime labels ('bull', 'bear', 'sideways')
        title: Chart title
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    equity = np.array([float(e) for e in equity_curve])

    # Align lengths (take minimum)
    min_len = min(len(equity), len(dates))
    equity = equity[:min_len]
    dates = dates[:min_len]

    returns = np.diff(equity) / equity[:-1]

    # Auto-detect regimes if not provided (60-day rolling)
    if regime_labels is None:
        regime_labels = _detect_market_regimes(returns, window=60)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # Equity curve with regime coloring
    ax1.semilogy(dates, equity, color=COLORS["equity"], linewidth=1.5)

    # Add regime background colors
    regime_colors = {
        "bull": (*plt.cm.Greens(0.3)[:3], 0.3),
        "bear": (*plt.cm.Reds(0.3)[:3], 0.3),
        "sideways": (*plt.cm.Greys(0.3)[:3], 0.2),
    }

    _add_regime_background(ax1, dates, regime_labels, regime_colors)

    ax1.set_ylabel("Equity (KRW, log scale)", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Regime performance summary
    regime_stats = _calculate_regime_stats(equity, regime_labels)
    _plot_regime_summary(ax2, regime_stats)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Market regime analysis saved to {save_path}")

    if show:
        plt.show()

    return fig


def _detect_market_regimes(returns: np.ndarray, window: int = 60) -> list[str]:
    """Auto-detect market regimes based on rolling returns.

    Args:
        returns: Array of daily returns
        window: Rolling window size

    Returns:
        List of regime labels
    """
    # Pad returns to match original length
    returns_padded = np.concatenate([[0], returns])

    df = pd.DataFrame({"returns": returns_padded})
    rolling_return = df["returns"].rolling(window=window, min_periods=1).sum() * 100

    regimes = []
    for ret in rolling_return:
        if ret > 10:
            regimes.append("bull")
        elif ret < -10:
            regimes.append("bear")
        else:
            regimes.append("sideways")

    return regimes


def _add_regime_background(
    ax: plt.Axes,
    dates: list[datetime],
    regimes: list[str],
    colors: dict[str, tuple],
) -> None:
    """Add background colors for different regimes."""
    if not regimes:
        return

    current_regime = regimes[0]
    start_idx = 0

    for i, regime in enumerate(regimes[1:], 1):
        if regime != current_regime:
            ax.axvspan(
                dates[start_idx],
                dates[i - 1],
                facecolor=colors.get(current_regime, colors["sideways"]),
                alpha=0.3,
            )
            current_regime = regime
            start_idx = i

    # Last segment
    ax.axvspan(
        dates[start_idx],
        dates[-1],
        facecolor=colors.get(current_regime, colors["sideways"]),
        alpha=0.3,
    )


def _calculate_regime_stats(equity: np.ndarray, regimes: list[str]) -> dict:
    """Calculate performance statistics per regime."""
    df = pd.DataFrame({"equity": equity, "regime": regimes})
    df["returns"] = df["equity"].pct_change().fillna(0)

    stats = {}
    for regime in ["bull", "bear", "sideways"]:
        regime_data = df[df["regime"] == regime]
        if len(regime_data) > 0:
            total_return = (1 + regime_data["returns"]).prod() - 1
            avg_daily = regime_data["returns"].mean()
            count = len(regime_data)
            stats[regime] = {
                "total_return": total_return * 100,
                "avg_daily": avg_daily * 100,
                "days": count,
            }
        else:
            stats[regime] = {"total_return": 0, "avg_daily": 0, "days": 0}

    return stats


def _plot_regime_summary(ax: plt.Axes, stats: dict) -> None:
    """Plot regime performance summary as horizontal bar chart."""
    regimes = ["bull", "bear", "sideways"]
    regime_names = ["Bull", "Bear", "Sideways"]
    colors_list = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]

    returns = [stats[r]["total_return"] for r in regimes]
    days = [stats[r]["days"] for r in regimes]

    y_pos = np.arange(len(regimes))

    bars = ax.barh(y_pos, returns, color=colors_list, edgecolor="white", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{name}\n({d} days)" for name, d in zip(regime_names, days, strict=False)])
    ax.set_xlabel("Cumulative Return (%)", fontsize=11)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels
    for bar, ret in zip(bars, returns, strict=False):
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 5 if width >= 0 else -5
        ax.annotate(
            f"{ret:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=10,
            fontweight="bold",
        )


def save_all_charts(
    metrics: PerformanceMetrics,
    wfa_results: list[dict] | None = None,
    output_dir: Path | None = None,
    prefix: str = "backtest",
) -> list[Path]:
    """Generate and save all visualization charts.

    Args:
        metrics: Performance metrics from backtest
        wfa_results: Optional WFA window results
        output_dir: Directory to save charts (defaults to 'output/')
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    if output_dir is None:
        output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Equity curve
    equity_path = output_dir / f"{prefix}_equity_curve.png"
    plot_equity_curve(
        metrics.equity_curve,
        metrics.dates,
        title=f"VBO Strategy - Equity Curve (CAGR: {float(metrics.cagr):.1f}%)",
        save_path=equity_path,
        show=False,
    )
    saved_files.append(equity_path)

    # Yearly returns
    if metrics.yearly_returns:
        yearly_path = output_dir / f"{prefix}_yearly_returns.png"
        plot_yearly_returns(
            metrics.yearly_returns,
            title="VBO Strategy - Yearly Returns",
            save_path=yearly_path,
            show=False,
        )
        saved_files.append(yearly_path)

    # Market regime analysis
    regime_path = output_dir / f"{prefix}_market_regime.png"
    plot_market_regime_analysis(
        metrics.equity_curve,
        metrics.dates,
        title="VBO Strategy - Market Regime Analysis",
        save_path=regime_path,
        show=False,
    )
    saved_files.append(regime_path)

    # WFA results
    if wfa_results:
        wfa_path = output_dir / f"{prefix}_wfa_results.png"
        plot_wfa_results(
            wfa_results,
            title="Walk Forward Analysis Results",
            save_path=wfa_path,
            show=False,
        )
        saved_files.append(wfa_path)

    logger.info(f"Saved {len(saved_files)} charts to {output_dir}")
    return saved_files
