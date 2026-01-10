"""Example: Run Walk Forward Analysis validation.

Demonstrates:
- Setting up WFA with custom parameters
- Creating backtest function
- Analyzing results across multiple windows
"""

from decimal import Decimal

import pandas as pd

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage
from bt.engine.backtest import BacktestEngine
from bt.logging import setup_logging
from bt.reporting.metrics import calculate_performance_metrics
from bt.strategies.allocation import create_cash_partition_allocator
from bt.strategies.vbo import get_vbo_strategy
from bt.validation.wfa import WalkForwardAnalysis


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load market data."""
    from pathlib import Path

    file_path = Path("data") / interval / f"{symbol}.parquet"
    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def backtest_on_data(data: dict[str, pd.DataFrame], params: dict | None = None) -> dict:
    """Run backtest on given data.

    Args:
        data: Dictionary of symbol -> DataFrame
        params: Optional parameters

    Returns:
        Dictionary with metrics
    """
    if params is None:
        params = {}

    # Create configuration
    config = BacktestConfig(
        initial_cash=Amount(Decimal(str(params.get("initial_cash", "10000000")))),
        fee=Fee(Decimal(str(params.get("fee", "0.0005")))),
        slippage=Percentage(Decimal(str(params.get("slippage", "0.0005")))),
        multiplier=params.get("multiplier", 2),
        lookback=params.get("lookback", 5),
        interval=params.get("interval", "day"),
    )

    # Initialize engine
    engine = BacktestEngine(config)

    # Load data
    symbols = list(data.keys())
    for symbol, df in data.items():
        engine.load_data(symbol, df)

    # Get strategy and run
    strategy = get_vbo_strategy()
    allocation_func = create_cash_partition_allocator(symbols)

    try:
        engine.run(
            symbols=symbols,
            buy_conditions=strategy["buy_conditions"],
            sell_conditions=strategy["sell_conditions"],
            buy_price_func=strategy["buy_price_func"],
            sell_price_func=strategy["sell_price_func"],
            allocation_func=allocation_func,
        )

        # Calculate metrics
        metrics = calculate_performance_metrics(
            equity_curve=engine.portfolio.equity_curve,
            dates=engine.portfolio.dates,
            trades=engine.portfolio.trades,
            _initial_cash=config.initial_cash,
        )

        return {
            "cagr": float(metrics.cagr),
            "mdd": float(metrics.mdd),
            "win_rate": float(metrics.win_rate),
            "sortino_ratio": float(metrics.sortino_ratio),
        }
    except Exception as e:
        print(f"Error in backtest: {e}")
        return {"cagr": 0, "mdd": 0, "win_rate": 0, "sortino_ratio": 0}


def run_wfa_validation(
    symbols: list[str] | None = None,
    interval: str = "day",
    train_periods: int = 365,
    test_periods: int = 90,
    step_periods: int = 90,
) -> None:
    """Run Walk Forward Analysis.

    Args:
        symbols: List of symbols to validate
        interval: Time interval
        train_periods: Training window size (bars)
        test_periods: Test window size (bars)
        step_periods: Step size (bars)
    """
    if symbols is None:
        symbols = ["BTC", "ETH", "XRP", "TRX"]

    print("\n" + "=" * 60)
    print("WALK FORWARD ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    all_data = {}
    for symbol in symbols:
        try:
            df = load_data(symbol, interval)
            all_data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {symbol}: Failed - {e}")
            return

    # Align data to minimum length
    if all_data:
        min_length = min(len(df) for df in all_data.values())
        all_data = {symbol: df.iloc[:min_length].copy() for symbol, df in all_data.items()}
        print(f"\n  Aligned to {min_length} bars (minimum across all symbols)")

    # Create WFA instance
    wfa = WalkForwardAnalysis(
        train_periods=train_periods,
        test_periods=test_periods,
        step_periods=step_periods,
        anchored=False,
    )

    # Run WFA
    results = wfa.run(data=all_data, backtest_func=backtest_on_data)

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("WFA SUMMARY")
    print("=" * 60)
    print(f"\n  Windows:         {summary['num_windows']}")
    print(f"  Average CAGR:    {summary['avg_cagr']:>10.2f}% (±{summary['std_cagr']:.2f}%)")
    print(f"  Median CAGR:     {summary['median_cagr']:>10.2f}%")
    print(f"  CAGR Range:      {summary['min_cagr']:>10.2f}% to {summary['max_cagr']:.2f}%")
    print(f"  Average MDD:     {summary['avg_mdd']:>10.2f}%")
    print(f"  Median MDD:      {summary['median_mdd']:>10.2f}%")
    print(f"  Worst MDD:       {summary['worst_mdd']:>10.2f}%")
    print(f"  Average Win Rate:{summary['avg_win_rate']:>9.2f}%")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point."""
    setup_logging(level="INFO", log_format="text")
    run_wfa_validation()


if __name__ == "__main__":
    main()
