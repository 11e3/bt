"""Optimized performance metrics calculation using vectorized operations.

Provides high-performance alternatives to the original metrics calculations
with significant speed improvements for large datasets.
"""

from datetime import datetime
from decimal import Decimal

import numpy as np

from bt.domain.models import PerformanceMetrics, Trade
from bt.domain.types import Amount, Percentage
from bt.utils.constants import ZERO
from bt.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_max_drawdown_optimized(equity: np.ndarray) -> float:
    """Optimized maximum drawdown calculation using numpy.

    Args:
        equity: Array of equity values

    Returns:
        Maximum drawdown as negative percentage
    """
    if len(equity) < 2:
        return 0.0

    # Vectorized peak tracking
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calculate_sortino_ratio_optimized(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Optimized Sortino ratio calculation with downside deviation.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    # Excess returns (daily)
    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate

    # Downside deviation (vectorized)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) <= 1:
        return 0.0

    downside_deviation = np.std(downside_returns) * np.sqrt(252)
    mean_excess_return = np.mean(excess_returns) * 252

    return mean_excess_return / downside_deviation if downside_deviation > 0 else 0.0


def calculate_sharpe_ratio_optimized(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Optimized Sharpe ratio calculation.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    if len(returns) <= 1:
        return 0.0

    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_calmar_ratio_optimized(equity: np.ndarray, days: float) -> float:
    """Optimized Calmar ratio calculation.

    Args:
        equity: Array of equity values
        days: Number of days in the period

    Returns:
        Calmar ratio
    """
    if len(equity) < 2 or days <= 0:
        return 0.0

    years = days / 365.25
    if years <= 0:
        return 0.0

    (equity[-1] / equity[0]) - 1
    cagr = ((equity[-1] / equity[0]) ** (1 / years)) - 1
    max_dd = abs(calculate_max_drawdown_optimized(equity))

    return cagr / max_dd if max_dd > 0 else 0.0


def calculate_var_optimized(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk using quantile method.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)

    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0

    return float(np.percentile(returns, confidence_level * 100))


def calculate_cvar_optimized(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Array of returns
        confidence_level: Confidence level

    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var_optimized(returns, confidence_level)
    tail_returns = returns[returns <= var]

    return float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0


def calculate_performance_metrics_optimized(
    equity_curve: list[Decimal],
    dates: list[datetime],
    trades: list[Trade],
    initial_cash: Amount,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics with optimizations.

    This is a drop-in replacement for calculate_performance_metrics
    with significant performance improvements for large datasets.

    Args:
        equity_curve: List of portfolio values over time
        dates: List of corresponding dates
        trades: List of completed trades
        initial_cash: Initial capital

    Returns:
        PerformanceMetrics with all calculated metrics
    """
    if len(equity_curve) < 2:
        logger.warning("Insufficient data for optimized metrics calculation")
        return _empty_metrics()

    # Convert to numpy arrays for vectorized operations
    equity = np.array([float(e) for e in equity_curve])

    # Total return (vectorized)
    total_return = Percentage(Decimal((equity[-1] / equity[0] - 1) * 100))

    # CAGR calculation (optimized)
    days = (dates[-1] - dates[0]).days
    years = days / 365.25
    cagr = Percentage(
        Decimal(((equity[-1] / equity[0]) ** (1 / years) - 1) * 100) if years > 0 else ZERO
    )

    # Maximum drawdown (optimized)
    mdd = Percentage(Decimal(calculate_max_drawdown_optimized(equity) * 100))

    # Returns calculation (vectorized)
    returns = np.diff(equity) / equity[:-1]

    # Risk metrics (optimized)
    sortino = Decimal(calculate_sortino_ratio_optimized(returns))
    sharpe = Decimal(calculate_sharpe_ratio_optimized(returns))
    calmar = Decimal(calculate_calmar_ratio_optimized(equity, days))

    # VaR and CVaR (new optimized metrics)
    var_95 = Percentage(Decimal(calculate_var_optimized(returns, 0.05) * 100))
    cvar_95 = Percentage(Decimal(calculate_cvar_optimized(returns, 0.05) * 100))

    # Trade statistics (vectorized where possible)
    if trades:
        pnl_values = np.array([float(t.pnl) for t in trades])
        np.array([float(t.return_pct) for t in trades])

        winning_trades = pnl_values > 0
        losing_trades = pnl_values < 0

        win_rate = Percentage(Decimal((np.sum(winning_trades) / len(trades)) * 100))

        # Profit factor
        total_wins = np.sum(pnl_values[winning_trades]) if np.any(winning_trades) else 0
        total_losses = abs(np.sum(pnl_values[losing_trades])) if np.any(losing_trades) else 1
        profit_factor = Decimal(total_wins / total_losses) if total_losses > 0 else Decimal("inf")

        # Average win/loss
        avg_win = Decimal(np.mean(pnl_values[winning_trades])) if np.any(winning_trades) else ZERO
        avg_loss = Decimal(np.mean(pnl_values[losing_trades])) if np.any(losing_trades) else ZERO

        # Best/worst trades
        best_trade = Decimal(np.max(pnl_values))
        worst_trade = Decimal(np.min(pnl_values))

    else:
        win_rate = ZERO
        profit_factor = ZERO
        avg_win = ZERO
        avg_loss = ZERO
        best_trade = ZERO
        worst_trade = ZERO

    # Yearly returns (vectorized)
    yearly_returns = _calculate_yearly_returns_vectorized(equity_curve, dates)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        mdd=mdd,
        sortino=sortino,
        sharpe=sharpe,
        calmar=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=best_trade,
        worst_trade=worst_trade,
        num_trades=len(trades),
        yearly_returns=yearly_returns,
        equity_curve=equity_curve,
        dates=dates,
        initial_cash=initial_cash,
    )


def _calculate_yearly_returns_vectorized(
    equity_curve: list[Decimal], dates: list[datetime]
) -> dict[str, Percentage]:
    """Calculate yearly returns using vectorized operations.

    Args:
        equity_curve: List of portfolio values
        dates: List of corresponding dates

    Returns:
        Dictionary mapping years to yearly returns
    """
    if not dates or len(equity_curve) < 2:
        return {}

    # Convert to numpy for efficient operations
    years = np.array([d.year for d in dates])
    unique_years = np.unique(years)

    yearly_returns = {}

    for year in unique_years:
        year_mask = years == year
        year_indices = np.where(year_mask)[0]

        if len(year_indices) >= 2:
            start_value = float(equity_curve[year_indices[0]])
            end_value = float(equity_curve[year_indices[-1]])

            if start_value > 0:
                yearly_return = (end_value / start_value - 1) * 100
                yearly_returns[str(int(year))] = Percentage(Decimal(yearly_return))

    return yearly_returns


def _empty_metrics() -> PerformanceMetrics:
    """Return empty PerformanceMetrics for edge cases."""
    return PerformanceMetrics(
        total_return=ZERO,
        cagr=ZERO,
        mdd=ZERO,
        sortino=ZERO,
        sharpe=ZERO,
        calmar=ZERO,
        var_95=ZERO,
        cvar_95=ZERO,
        win_rate=ZERO,
        profit_factor=ZERO,
        avg_win=ZERO,
        avg_loss=ZERO,
        best_trade=ZERO,
        worst_trade=ZERO,
        num_trades=0,
        yearly_returns={},
        equity_curve=[],
        dates=[],
        initial_cash=Amount(ZERO),
    )
