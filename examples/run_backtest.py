"""Composable Backtest Runner.

Strategies are composed dynamically from conditions, pricing, and allocation modules.
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

# Import components
from bt.strategies import allocation, conditions, pricing


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
    # -------------------------------------------------------------------------
    # [사용자 설정]
    # -------------------------------------------------------------------------
    strategy_name = "vbo"  # "vbo" or "bnh"
    symbols = ["BTC", "ETH", "XRP", "TRX"]
    interval = "day"
    initial_cash = 10_000_000
    fee = 0.0005
    slippage = 0.0005
    multiplier = 2
    lookback = 5
    # -------------------------------------------------------------------------

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

    # 3. Strategy Composition (핵심: 여기서 조립)
    # 각 전략이 어떤 부품을 쓰는지 여기서 정의합니다.
    strategy_map = {
        "vbo": {
            "buy_conditions": {
                "no_pos": conditions.no_open_position,
                "breakout": conditions.vbo_breakout_triggered,
                "trend_short": conditions.price_above_short_ma,
                "trend_long": conditions.price_above_long_ma,
            },
            "sell_conditions": {
                "has_pos": conditions.has_open_position,
                "stop_trend": conditions.close_below_short_ma,
            },
            "buy_price_func": pricing.get_vbo_buy_price,
            "sell_price_func": pricing.get_current_close,
            # VBO는 다종목 분산 투자 (Cash Partition)
            "allocation_func": allocation.create_cash_partition_allocator(loaded_symbols),
        },
        "bnh": {
            "buy_conditions": {
                "no_pos": conditions.no_open_position,
            },
            "sell_conditions": {
                "never": conditions.never,
            },
            "buy_price_func": pricing.get_current_close,
            "sell_price_func": pricing.get_current_close,
            # BnH는 단일 종목 올인 (All-in)
            "allocation_func": allocation.all_in_allocation,
        },
    }

    if strategy_name.lower() not in strategy_map:
        logger.error(f"Unknown strategy: {strategy_name}")
        return

    strategy = strategy_map[strategy_name.lower()]

    logger.info(f"Running {strategy_name.upper()} Backtest")

    # 4. Run
    engine.run(
        symbols=loaded_symbols,
        buy_conditions=strategy["buy_conditions"],
        sell_conditions=strategy["sell_conditions"],
        buy_price_func=strategy["buy_price_func"],
        sell_price_func=strategy["sell_price_func"],
        allocation_func=strategy["allocation_func"],
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
