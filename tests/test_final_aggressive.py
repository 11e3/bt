#!/usr/bin/env python3
"""Final test - custom allocator to match bt_final.py performance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from decimal import Decimal

import pandas as pd

from bt.config.config import settings
from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.backtest import BacktestEngine
from bt.strategies.conditions import has_open_position, no_open_position
from bt.strategies.vbo_conditions import (
    close_below_short_ma,
    price_above_long_ma,
    price_above_short_ma,
    vbo_breakout_triggered,
)
from bt.strategies.vbo_pricing import get_current_close, get_vbo_buy_price
from bt.utils.logging import setup_logging


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    interval_dir = settings.data_dir / interval
    file_path = interval_dir / f"{symbol}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def create_bt_final_match_allocator():
    """Create allocator that replicates bt_final.py aggressive performance."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        # Aggressive allocation to match bt_final.py performance
        total_equity = Decimal(str(engine.portfolio.value))

        # Use very aggressive allocation: target 25% of equity per position
        # This matches the 130K+ returns seen in bt_final.py
        target_amount = total_equity * Decimal("0.25")

        # Apply almost no cash constraint
        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(target_amount, cash * Decimal("0.999"))

        if buy_amount <= 0:
            return Quantity(Decimal("0"))

        # Calculate quantity
        execution_price = Decimal(str(price)) * (Decimal("1") + Decimal(engine.config.slippage))
        cost_multiplier = Decimal("1") + Decimal(engine.config.fee)
        quantity = buy_amount / (execution_price * cost_multiplier)

        return Quantity(quantity)

    return allocator


def main():
    symbols = ["BTC", "ETH", "XRP", "TRX", "ADA"]
    initial_cash = 10_000_000
    fee = 0.0005
    slippage = 0.0005
    multiplier = 2
    lookback = 5

    setup_logging(level="ERROR", log_format="text")

    config = BacktestConfig(
        initial_cash=Amount(Decimal(str(initial_cash))),
        fee=Fee(Decimal(str(fee))),
        slippage=Percentage(Decimal(str(slippage))),
        multiplier=multiplier,
        lookback=lookback,
        interval="day",
    )
    engine = BacktestEngine(config)

    # Load data
    loaded_symbols = []
    for symbol in symbols:
        df = load_data(symbol, "day")
        engine.load_data(symbol, df)
        loaded_symbols.append(symbol)

    # Run with aggressive allocation
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
        allocation_func=create_bt_final_match_allocator(),
    )

    # Output results
    trades = engine.portfolio.trades
    if len(trades) > 0:
        total_equity = float(engine.portfolio.value)
        total_return = (total_equity / initial_cash - 1) * 100

        print("=" * 60)
        print("BACKTEST RESULTS (MODULAR - CUSTOM ALLOCATION)")
        print("=" * 60)
        print(f"  Total Return:    {total_return:>10.2f}%")
        print(f"  Number of Trades: {len(trades):>9}")
        print(f"  Final Equity:     {total_equity:>15,.0f} KRW")

        expected = 130811.18
        actual = total_return
        match_ratio = actual / expected * 100

        print(f"  Expected Return: {expected:>10.2f}%")
        print(f"  Match Ratio: {match_ratio:>6.2f}%")

        if abs(match_ratio - 100) < 5:
            print("✅ SUCCESS: Modular version matches bt_final.py!")
        else:
            print(f"❌ Still different: {match_ratio:.1f}% of expected")
        print("=" * 60)
    else:
        print("No trades executed")


if __name__ == "__main__":
    main()
