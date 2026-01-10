"""Example: Run VBO strategy backtest with new architecture.

Demonstrates:
- Loading data with DataProvider
- Configuring strategy with Protocols
- Running backtest with dependency injection
- Calculating and displaying metrics
"""

from decimal import Decimal

import pandas as pd

from bt.config import settings
from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage
from bt.engine.backtest import BacktestEngine
from bt.logging import get_logger, setup_logging
from bt.reporting.metrics import (
    calculate_performance_metrics,
    print_performance_report,
    print_sample_trades,
)
from bt.strategies.allocation import create_cash_partition_allocator
from bt.strategies.vbo import get_vbo_strategy


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load parquet data for a symbol.

    Args:
        symbol: Trading symbol
        interval: Time interval

    Returns:
        DataFrame with OHLCV data

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    interval_dir = settings.data_dir / interval
    file_path = interval_dir / f"{symbol}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Validate required columns
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return df


def run_backtest(
    symbols: list[str] | None = None,
    interval: str = "day",
    initial_cash: Decimal = Decimal("10000000"),
    fee: Decimal = Decimal("0.0005"),
    slippage: Decimal = Decimal("0.0005"),
    multiplier: int = 2,
    lookback: int = 5,
) -> None:
    """Run VBO strategy backtest.

    Args:
        symbols: List of symbols to trade
        interval: Time interval
        initial_cash: Initial capital in KRW
        fee: Trading fee (0.0005 = 0.05%)
        slippage: Slippage (0.0005 = 0.05%)
        multiplier: Multiplier for long-term indicators
        lookback: Lookback period for short-term indicators
    """
    if symbols is None:
        symbols = ["BTC", "ETH", "XRP", "TRX"]

    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("VBO STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(
        "Configuration",
        extra={
            "symbols": symbols,
            "interval": interval,
            "initial_cash": float(initial_cash),
            "fee": float(fee),
            "slippage": float(slippage),
            "multiplier": multiplier,
            "lookback": lookback,
        },
    )

    # Create configuration
    config = BacktestConfig(
        initial_cash=Amount(initial_cash),
        fee=Fee(fee),
        slippage=Percentage(slippage),
        multiplier=multiplier,
        lookback=lookback,
        interval=interval,
    )

    # Initialize engine with dependency injection
    engine = BacktestEngine(config)

    # Load data for all symbols
    logger.info("Loading market data...")
    for symbol in symbols:
        try:
            df = load_data(symbol, interval)
            engine.load_data(symbol, df)
            logger.info(
                f"Loaded {symbol}",
                extra={
                    "rows": len(df),
                    "start": df["datetime"].min().isoformat(),
                    "end": df["datetime"].max().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to load {symbol}", extra={"error": str(e)})
            return

    # Get strategy configuration
    strategy = get_vbo_strategy()

    # Create allocation function
    allocation_func = create_cash_partition_allocator(symbols)

    # Run backtest
    logger.info("Running backtest...")
    engine.run(
        symbols=symbols,
        buy_conditions=strategy["buy_conditions"],
        sell_conditions=strategy["sell_conditions"],
        buy_price_func=strategy["buy_price_func"],
        sell_price_func=strategy["sell_price_func"],
        allocation_func=allocation_func,
    )

    # Calculate performance metrics
    logger.info("Calculating performance metrics...")
    metrics = calculate_performance_metrics(
        equity_curve=engine.portfolio.equity_curve,
        dates=engine.portfolio.dates,
        trades=engine.portfolio.trades,
        _initial_cash=config.initial_cash,
    )

    # Display results
    print_performance_report(metrics)
    print_sample_trades(metrics.trades, max_trades=10)


def main() -> None:
    """Main entry point."""
    # Setup logging
    setup_logging(level="INFO", log_format="text")

    # Run backtest with default parameters
    run_backtest()


if __name__ == "__main__":
    main()
