"""Performance metrics calculation and reporting.

Calculates comprehensive backtest performance metrics including
returns, risk measures, and trade statistics.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from bt.domain.models import PerformanceMetrics, Trade
from bt.domain.types import Amount, Percentage, Price, Quantity
from bt.utils.constants import METRIC_PRECISION, ZERO
from bt.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_performance_metrics(
    equity_curve: list[Decimal],
    dates: list[datetime],
    trades: list[Trade],
    _initial_cash: Amount,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        equity_curve: List of portfolio values over time
        dates: List of corresponding dates
        trades: List of completed trades
        initial_cash: Initial capital

    Returns:
        PerformanceMetrics with all calculated metrics

    Why separate function:
    - Pure calculation logic for easy testing
    - No side effects
    - Reusable across different contexts
    """
    if len(equity_curve) < 2:
        logger.warning("Insufficient data for metrics calculation")
        return _empty_metrics()

    # Convert to numpy array for calculations
    equity = np.array([float(e) for e in equity_curve])

    # Total return
    total_return = Percentage(Decimal((equity[-1] / equity[0] - 1) * 100))

    # Calculate CAGR
    days = (dates[-1] - dates[0]).days
    years = days / 365.25
    cagr = Percentage(
        Decimal(((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0)
    )

    # Calculate Maximum Drawdown
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    mdd = Percentage(Decimal(drawdown.min() * 100))

    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    # Use 365 for crypto (trades every day) instead of 252 for stocks
    annualized_mean = mean_return * 365

    # Sharpe Ratio (using total standard deviation)
    total_std = np.std(returns) if len(returns) > 1 else 0
    annualized_std = total_std * np.sqrt(365) if total_std > 0 else 0
    if np.isnan(annualized_std) or annualized_std == 0 or np.isnan(annualized_mean):
        sharpe = Decimal(0)
    else:
        sharpe = Decimal(annualized_mean / annualized_std)

    # Sortino Ratio (using downside deviation only)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
    annualized_downside_std = downside_std * np.sqrt(365) if downside_std > 0 else 0
    if (
        np.isnan(annualized_downside_std)
        or annualized_downside_std == 0
        or np.isnan(annualized_mean)
    ):
        sortino = Decimal(0)
    else:
        sortino = Decimal(annualized_mean / annualized_downside_std)

    # Trade statistics
    # Support both Trade objects and dicts
    def get_pnl(t):
        """Extract PnL from Trade object or dict."""
        if hasattr(t, "pnl"):
            return float(t.pnl)
        if isinstance(t, dict):
            return float(t.get("pnl", 0))
        return 0

    if trades:
        # Filter trades that have PnL data
        trades_with_pnl = [t for t in trades if get_pnl(t) != 0 or hasattr(t, "pnl")]

        if trades_with_pnl:
            winning_trades = [t for t in trades_with_pnl if get_pnl(t) > 0]
            losing_trades = [t for t in trades_with_pnl if get_pnl(t) <= 0]

            win_rate = Percentage(Decimal(len(winning_trades) / len(trades_with_pnl) * 100))

            total_profit = sum(get_pnl(t) for t in winning_trades)
            total_loss = abs(sum(get_pnl(t) for t in losing_trades))
            # Avoid infinity by capping very large values
            if total_loss != 0:
                pf = Decimal(total_profit / total_loss)
                profit_factor = min(pf, METRIC_PRECISION)  # Cap at large finite number
            else:
                profit_factor = METRIC_PRECISION  # All winning trades: cap at large number

            avg_win = Amount(
                Decimal(str(float(np.mean([get_pnl(t) for t in winning_trades]))))
                if winning_trades
                else ZERO
            )
            avg_loss = Amount(
                Decimal(str(float(np.mean([get_pnl(t) for t in losing_trades]))))
                if losing_trades
                else ZERO
            )
        else:
            # No trades with PnL data
            win_rate = Percentage(ZERO)
            profit_factor = ZERO
            avg_win = Amount(ZERO)
            avg_loss = Amount(ZERO)
    else:
        win_rate = Percentage(ZERO)
        profit_factor = ZERO
        avg_win = Amount(ZERO)
        avg_loss = Amount(ZERO)

    # Yearly returns
    yearly_returns = calculate_yearly_returns(equity, dates)

    # Convert dict trades to Trade objects, keep existing Trade objects
    valid_trades: list[Trade] = []
    for t in trades:
        if isinstance(t, Trade):
            valid_trades.append(t)
        elif isinstance(t, dict):
            try:
                # Convert dict to Trade object
                entry_date = t.get("date", t.get("entry_date", datetime.now(tz=timezone.utc)))

                if isinstance(entry_date, str):
                    entry_date = datetime.fromisoformat(entry_date)
                exit_date = t.get("exit_date", entry_date)
                if isinstance(exit_date, str):
                    exit_date = datetime.fromisoformat(exit_date)

                trade = Trade(
                    symbol=t.get("symbol", "UNKNOWN"),
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=Price(Decimal(str(t.get("entry_price", t.get("price", 0))))),
                    exit_price=Price(Decimal(str(t.get("exit_price", t.get("price", 0))))),
                    quantity=Quantity(Decimal(str(abs(t.get("quantity", 1))))),
                    pnl=Amount(Decimal(str(t.get("pnl", 0)))),
                    return_pct=Percentage(Decimal(str(t.get("return_pct", 0)))),
                )
                valid_trades.append(trade)
            except Exception as e:
                logger.debug(f"Could not convert trade dict to Trade object: {e}")

    metrics = PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        mdd=mdd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        avg_win=avg_win,
        avg_loss=avg_loss,
        final_equity=Amount(Decimal(equity[-1])),
        equity_curve=equity_curve,
        dates=dates,
        trades=valid_trades,
        yearly_returns=yearly_returns,
    )

    logger.info(
        "Performance metrics calculated",
        extra={
            "cagr": float(cagr),
            "mdd": float(mdd),
            "win_rate": float(win_rate),
            "num_trades": len(trades),
        },
    )

    return metrics


def calculate_yearly_returns(
    equity: np.ndarray[Any, Any],
    dates: list[datetime],
) -> dict[int, Percentage]:
    """Calculate returns for each year.

    Args:
        equity: Array of portfolio values
        dates: List of dates

    Returns:
        Dictionary mapping year to return percentage
    """
    # [수정] 데이터 길이 불일치 방지 (짧은 쪽에 맞춤)
    min_len = min(len(equity), len(dates))

    if min_len == 0:
        return {}

    # 앞부분부터 min_len만큼만 사용
    use_equity = equity[:min_len]
    use_dates = dates[:min_len]

    yearly_returns = {}

    # Convert dates to years
    years = [d.year for d in use_dates]
    unique_years = sorted(set(years))

    for year in unique_years:
        year_indices = [i for i, y in enumerate(years) if y == year]
        if len(year_indices) > 1:
            start_idx = year_indices[0]
            end_idx = year_indices[-1]
            start_equity = use_equity[start_idx]
            end_equity = use_equity[end_idx]
            year_return = Percentage(Decimal((end_equity / start_equity - 1) * 100))
            yearly_returns[year] = year_return

    return yearly_returns


def _empty_metrics() -> PerformanceMetrics:
    """Create empty metrics for edge cases."""
    return PerformanceMetrics(
        total_return=Percentage(ZERO),
        cagr=Percentage(ZERO),
        mdd=Percentage(ZERO),
        sharpe_ratio=ZERO,
        sortino_ratio=ZERO,
        win_rate=Percentage(ZERO),
        profit_factor=ZERO,
        num_trades=0,
        avg_win=Amount(ZERO),
        avg_loss=Amount(ZERO),
        final_equity=Amount(ZERO),
    )


def print_performance_report(metrics: PerformanceMetrics) -> None:
    """Print formatted performance report.

    Args:
        metrics: Performance metrics to display
    """
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\nPerformance Metrics:")
    print(f"  Total Return:    {float(metrics.total_return):>10.2f}%")
    print(f"  CAGR:            {float(metrics.cagr):>10.2f}%")
    print(f"  MDD:             {float(metrics.mdd):>10.2f}%")
    print(f"  Sharpe Ratio:    {float(metrics.sharpe_ratio):>10.2f}")
    print(f"  Sortino Ratio:   {float(metrics.sortino_ratio):>10.2f}")

    print("\nTrade Statistics:")
    print(f"  Number of Trades: {metrics.num_trades:>9}")
    print(f"  Win Rate:         {float(metrics.win_rate):>9.2f}%")
    print(f"  Profit Factor:    {float(metrics.profit_factor):>9.2f}")
    print(f"  Avg Win:          {float(metrics.avg_win):>9,.0f} KRW")
    print(f"  Avg Loss:         {float(metrics.avg_loss):>9,.0f} KRW")

    print("\nYearly Returns:")
    for year, ret in sorted(metrics.yearly_returns.items()):
        print(f"  {year}:  {float(ret):>10.2f}%")

    print(f"\nFinal Equity:     {float(metrics.final_equity):>15,.0f} KRW")
    print("=" * 60 + "\n")


def print_sample_trades(trades: list[Trade], max_trades: int = 10) -> None:
    """Print sample trades in formatted table.

    Args:
        trades: List of trades
        max_trades: Maximum number of trades to display
    """
    if not trades:
        return

    print("\n" + "=" * 60)
    print(f"SAMPLE TRADES (First {min(len(trades), max_trades)})")
    print("=" * 60)
    print(
        f"{'Symbol':<6} {'Entry Date':<12} {'Exit Date':<12} "
        f"{'Entry':<10} {'Exit':<10} {'Return':<8} {'P&L':<12}"
    )
    print("-" * 60)

    for trade in trades[:max_trades]:
        print(
            f"{trade.symbol:<6} "
            f"{trade.entry_date.strftime('%Y-%m-%d'):<12} "
            f"{trade.exit_date.strftime('%Y-%m-%d'):<12} "
            f"{float(trade.entry_price):<10,.0f} "
            f"{float(trade.exit_price):<10,.0f} "
            f"{float(trade.return_pct):>6.2f}% "
            f"{float(trade.pnl):>11,.0f}"
        )

    if len(trades) > max_trades:
        print(f"... and {len(trades) - max_trades} more trades")
    print("=" * 60)
