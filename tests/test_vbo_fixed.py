#!/usr/bin/env python3
"""Test VBO fixed strategy using modular components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decimal import Decimal

import pandas as pd

from bt.config.config import settings
from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage
from bt.engine.backtest import BacktestEngine
from bt.reporting.metrics import (
    calculate_performance_metrics,
    print_performance_report,
    print_sample_trades,
)
from bt.strategies.conditions import has_open_position, no_open_position

# Import VBO fixed components
from bt.strategies.vbo_allocation import create_vbo_momentum_allocator
from bt.strategies.vbo_conditions import (
    close_below_short_ma,
    price_above_long_ma,
    price_above_short_ma,
    vbo_breakout_triggered,
)
from bt.strategies.vbo_pricing import get_current_close, get_vbo_buy_price
from bt.utils.logging import get_logger, setup_logging


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load parquet data."""
    interval_dir = settings.data_dir / interval
    file_path = interval_dir / f"{symbol}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def main() -> None:
    # Configuration (match bt_fixed.py)
    symbols = ["BTC", "ETH", "XRP", "TRX", "ADA"]
    top_n = 5
    mom_lookback = 15
    interval = "day"
    initial_cash = 10_000_000
    fee = 0.0005
    slippage = 0.0005
    multiplier = 2
    lookback = 5

    setup_logging(level="INFO", log_format="text")
    logger = get_logger(__name__)

    # 1. Config & Engine
    config = BacktestConfig(
        initial_cash=Amount(Decimal(str(initial_cash))),
        fee=Fee(Decimal(str(fee))),
        slippage=Percentage(Decimal(str(slippage))),
        multiplier=multiplier,
        lookback=lookback,
        interval=interval,
    )
    engine = BacktestEngine(config)

    # 2. Load Data
    logger.info("Loading data...")
    loaded_symbols = []
    for symbol in symbols:
        try:
            df = load_data(symbol, interval)
            engine.load_data(symbol, df)
            loaded_symbols.append(symbol)
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    if not loaded_symbols:
        return

    # 3. VBO Fixed Strategy
    logger.info("Running VBO Fixed Backtest")

    # 4. Run
    engine.run(
        symbols=loaded_symbols,
        buy_conditions={
            "no_pos": no_open_position,
            "breakout": vbo_breakout_triggered,
            "trend_short": price_above_short_ma,
            "trend_long": price_above_long_ma,
        },
        sell_conditions={
            "has_pos": has_open_position,
            "stop_trend": close_below_short_ma,
        },
        buy_price_func=get_vbo_buy_price,
        sell_price_func=get_current_close,
        allocation_func=create_vbo_momentum_allocator(top_n=top_n, mom_lookback=mom_lookback),
    )

    # 5. Metrics
    metrics = calculate_performance_metrics(
        equity_curve=engine.portfolio.equity_curve,
        dates=engine.portfolio.dates,
        trades=engine.portfolio.trades,
        _initial_cash=config.initial_cash,
    )

    print_performance_report(metrics)
    print_sample_trades(metrics.trades)


if __name__ == "__main__":
    main()
