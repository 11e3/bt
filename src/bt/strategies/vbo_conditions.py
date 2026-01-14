"""VBO strategy conditions with look-ahead bias prevention."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def vbo_breakout_triggered(engine: "BacktestEngine", symbol: str) -> bool:
    """Check if price breaks out above VBO level."""
    from bt.strategies.vbo_pricing import get_vbo_buy_price

    buy_price = get_vbo_buy_price(engine, symbol)
    current_bar = engine.get_bar(symbol)

    if current_bar is None or buy_price == 0:
        return False

    return Decimal(str(current_bar["high"])) >= Decimal(str(buy_price))


def price_above_short_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """Check if price is above short-term moving average."""
    lookback = engine.config.lookback
    bars = engine.get_bars(symbol, lookback + 1)

    if bars is None or len(bars) < lookback + 1:
        return False

    # Buy price should be above MA of previous closes
    close_series = bars["close"].iloc[:-1]
    close_sma = close_series.tail(lookback).mean()

    # Get current buy price
    from bt.strategies.vbo_pricing import get_vbo_buy_price

    buy_price = get_vbo_buy_price(engine, symbol)

    return Decimal(str(buy_price)) > Decimal(str(close_sma))


def price_above_long_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """Check if price is above long-term moving average."""
    lookback = engine.config.lookback
    multiplier = engine.config.multiplier
    long_lookback = multiplier * lookback

    bars = engine.get_bars(symbol, long_lookback + 1)
    if bars is None or len(bars) < long_lookback + 1:
        return False

    close_series = bars["close"].iloc[:-1]
    close_sma_long = close_series.tail(long_lookback).mean()

    from bt.strategies.vbo_pricing import get_vbo_buy_price

    buy_price = get_vbo_buy_price(engine, symbol)

    return Decimal(str(buy_price)) > Decimal(str(close_sma_long))


def close_below_short_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """Check if previous close was below short MA."""
    lookback = engine.config.lookback

    # Calculate MA using historical data only
    bars = engine.get_bars(symbol, lookback + 2)
    if bars is None or len(bars) < lookback + 2:
        return False

    # Previous bar (last completed bar)
    prev_bar = bars.iloc[-1]
    # MA of closes excluding previous bar
    close_series = bars["close"].iloc[:-1]
    close_sma = close_series.tail(lookback).mean()

    return Decimal(str(prev_bar["close"])) < Decimal(str(close_sma))
