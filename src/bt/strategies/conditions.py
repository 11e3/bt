"""Trading entry and exit conditions."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.strategies.indicators import calculate_noise_ratio
from bt.strategies.pricing import get_vbo_buy_price
from bt.utils.indicator_cache import get_indicator_cache

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine


# --- Common Conditions ---


def no_open_position(engine: "BacktestEngine", symbol: str) -> bool:
    if engine.portfolio is None:
        return False
    position = engine.portfolio.get_position(symbol)
    return not position.is_open


def has_open_position(engine: "BacktestEngine", symbol: str) -> bool:
    return not no_open_position(engine, symbol)


def never(engine: "BacktestEngine", symbol: str) -> bool:
    return False


# --- Trend Conditions ---


def price_above_short_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """VBO: Buy Price > Short-term SMA of Close."""
    lookback = engine.config.lookback
    bars = engine.get_bars(symbol, lookback + 1)

    if bars is None or len(bars) < lookback + 1:
        return False

    # Use optimized indicator caching
    cache = get_indicator_cache()
    close_series = bars["close"].iloc[:-1]  # Exclude current bar

    # Get cached SMA
    close_sma = cache.calculate_indicator(symbol, "sma", lookback, close_series)

    buy_price = get_vbo_buy_price(engine, symbol)

    return Decimal(buy_price) > Decimal(str(close_sma))


def price_above_long_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """VBO: Buy Price > Long-term SMA of Close."""
    lookback = engine.config.lookback
    multiplier = engine.config.multiplier
    long_lookback = multiplier * lookback

    bars = engine.get_bars(symbol, long_lookback + 1)
    if bars is None or len(bars) < long_lookback + 1:
        return False

    # Use optimized indicator caching
    cache = get_indicator_cache()
    close_series = bars["close"].iloc[:-1]
    close_sma_long = cache.calculate_indicator(symbol, "sma", long_lookback, close_series)

    buy_price = get_vbo_buy_price(engine, symbol)

    return Decimal(buy_price) > Decimal(str(close_sma_long))


def close_below_short_ma(engine: "BacktestEngine", symbol: str) -> bool:
    """Sell: Current Close < Short-term SMA."""
    lookback = engine.config.lookback
    current_bar = engine.get_bar(symbol)

    if current_bar is None:
        return False

    bars = engine.get_bars(symbol, lookback + 1)
    if bars is None or len(bars) < lookback + 1:
        return False

    # Use optimized indicator caching
    cache = get_indicator_cache()
    close_series = bars["close"].iloc[:-1]
    close_sma = cache.calculate_indicator(symbol, "sma", lookback, close_series)

    return Decimal(str(current_bar["close"])) < Decimal(str(close_sma))


# --- VBO Specific Conditions ---


def vbo_breakout_triggered(engine: "BacktestEngine", symbol: str) -> bool:
    current_bar = engine.get_bar(symbol)
    if current_bar is None:
        return False

    buy_price = get_vbo_buy_price(engine, symbol)
    return Decimal(str(current_bar["high"])) >= Decimal(buy_price)


def noise_is_decreasing(engine: "BacktestEngine", symbol: str) -> bool:
    """Checks if Short-term Noise MA < Long-term Noise MA."""
    lookback = engine.config.lookback
    multiplier = engine.config.multiplier
    long_lookback = multiplier * lookback

    bars = engine.get_bars(symbol, long_lookback + 1)
    if bars is None or len(bars) < long_lookback + 1:
        return False

    # 1. 지표 계산: 노이즈
    noise_series = calculate_noise_ratio(bars.iloc[:-1])

    # 2. 지표 계산: 단기/장기 이동평균
    # calculate_sma 함수를 써도 되지만, 여기선 시리즈의 끝부분 평균값만 필요하므로 tail().mean()이 효율적
    noise_sma_short = noise_series.tail(lookback).mean()
    noise_sma_long = noise_series.tail(long_lookback).mean()

    return bool(noise_sma_short < noise_sma_long)
