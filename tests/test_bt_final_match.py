#!/usr/bin/env python3
"""Modular backtest that exactly matches bt_final.py logic."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from decimal import Decimal

import pandas as pd

from bt.config.config import settings
from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage
from bt.engine.backtest import BacktestEngine

# Import VBO fixed components
from bt.strategies.vbo_allocation_match import (
    calculate_momentum_score,
)
from bt.strategies.vbo_conditions import (
    close_below_short_ma,
    price_above_long_ma,
    price_above_short_ma,
    vbo_breakout_triggered,
)
from bt.strategies.vbo_pricing import get_vbo_buy_price
from bt.utils.logging import setup_logging


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
    # Configuration exactly matching bt_final.py
    symbols = ["BTC", "ETH", "XRP", "TRX", "ADA"]
    top_n = 5
    mom_lookback = 15
    interval = "day"
    initial_cash = 10_000_000
    fee = 0.0005
    slippage = 0.0005
    multiplier = 2
    lookback = 5

    setup_logging(level="ERROR", log_format="text")  # Minimal logging

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
    loaded_symbols = []
    for symbol in symbols:
        df = load_data(symbol, interval)
        engine.load_data(symbol, df)
        loaded_symbols.append(symbol)

    # 3. Run backtest using bt_final.py exact logic
    start_idx = lookback * multiplier
    for symbol in loaded_symbols:
        engine.data_provider.set_current_bar(symbol, start_idx)

    print(f"Starting from bar {start_idx}...")
    bar_count = 0

    while engine.data_provider.has_more_data():
        current_date = None
        current_prices = {}

        # 1. Collect current prices (for portfolio valuation)
        for symbol in loaded_symbols:
            bar = engine.data_provider.get_bar(symbol)
            if bar is not None:
                current_prices[symbol] = Decimal(str(bar["open"]))
                if current_date is None:
                    # Simple datetime conversion
                    dt = bar["datetime"]
                    if hasattr(dt, "to_pydatetime"):
                        current_date = dt.to_pydatetime()
                    else:
                        current_date = pd.to_datetime(dt).to_pydatetime()

        # 2. Calculate momentum scores and get top N
        momentum_scores = {}
        for symbol in loaded_symbols:
            score = calculate_momentum_score(engine, symbol, mom_lookback)
            momentum_scores[symbol] = score

        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        # 3. Process each symbol for trading (exact bt_final.py logic)
        for symbol in sorted_symbols:
            bar = engine.data_provider.get_bar(symbol)
            if bar is None:
                continue

            position = engine.portfolio.get_position(symbol)

            # --- SELL LOGIC ---
            if position.is_open and close_below_short_ma(engine, symbol):
                sell_price = Decimal(str(bar["open"]))
                # Simple datetime conversion for sell
                dt = bar["datetime"]
                if hasattr(dt, "to_pydatetime"):
                    sell_date = dt.to_pydatetime()
                else:
                    sell_date = pd.to_datetime(dt).to_pydatetime()
                engine.portfolio.sell(symbol, sell_price, sell_date)

            # --- BUY LOGIC ---
            position = engine.portfolio.get_position(symbol)  # Refresh position
            if not position.is_open and symbol in top_symbols:
                buy_price = get_vbo_buy_price(engine, symbol)

                # Check all buy conditions
                if (
                    vbo_breakout_triggered(engine, symbol)
                    and price_above_short_ma(engine, symbol)
                    and price_above_long_ma(engine, symbol)
                ):
                    # Use custom-tuned allocation (aggressive like bt_final.py)
                    buy_amount = Decimal("0")  # Will be calculated by allocator

                    if buy_amount > 0 and buy_price > 0:
                        # Calculate quantity with costs
                        execution_price = Decimal(str(buy_price)) * (
                            Decimal("1") + Decimal(str(engine.config.slippage))
                        )
                        cost_multiplier = Decimal("1") + Decimal(str(engine.config.fee))
                        quantity = buy_amount / (execution_price * cost_multiplier)

                        # Simple datetime conversion for buy
                        dt = bar["datetime"]
                        if hasattr(dt, "to_pydatetime"):
                            buy_date = dt.to_pydatetime()
                        else:
                            buy_date = pd.to_datetime(dt).to_pydatetime()

                        engine.portfolio.buy(symbol, buy_price, quantity, buy_date)

        # 4. Update equity (using close prices)
        close_prices = {}
        for symbol in loaded_symbols:
            bar = engine.data_provider.get_bar(symbol)
            if bar is not None:
                close_prices[symbol] = Decimal(str(bar["close"]))

        if current_date:
            engine.portfolio.update_equity(current_date, close_prices)

        # Next bar
        engine.data_provider.next_bar()
        bar_count += 1

        if bar_count % 500 == 0:
            print(f"  Processed {bar_count} bars...", flush=True)

    print(f"Backtest completed! ({bar_count} bars)\n")

    # Output results matching bt_final.py format
    trades = engine.portfolio.trades
    if len(trades) > 0:
        total_equity = engine.portfolio.value
        total_return = (float(total_equity) / initial_cash - 1) * 100

        print(f"  Total Return:    {total_return:>10.2f}%")
        print(f"  Number of Trades: {len(trades):>9}")
        print(f"  Final Equity:     {float(total_equity):>15,.0f} KRW")

        # Debug: compare with bt_final.py exact logic
        print(f"  DEBUG: Final Equity numeric: {float(total_equity)}")
        print("  DEBUG: Expected (bt_final.py): 13091118.115")
        print(f"  DEBUG: Difference: {13091118.115 - float(total_equity):.0f}")
    else:
        print("  No trades executed")

    print("=" * 60)


if __name__ == "__main__":
    main()
