"""VBO (Volatility Breakout) pricing strategies."""

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from bt.domain.types import Price
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def calculate_noise_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculate noise ratio: |Open - Close| / (High - Low)."""
    numerator = (df["open"] - df["close"]).abs()
    denominator = df["high"] - df["low"]
    noise = numerator / denominator.replace(0, float("nan"))
    return noise.fillna(0)


def get_vbo_buy_price(engine: "BacktestEngine", symbol: str) -> Price:
    """Calculate VBO breakout buy price (look-ahead bias free)."""
    current_bar = engine.get_bar(symbol)
    if current_bar is None:
        return Price(Decimal("0"))

    # Get historical bars only (exclude current bar)
    lookback = engine.config.lookback
    bars = engine.get_bars(symbol, lookback + 1)

    if bars is None or len(bars) < lookback + 1:
        return Price(Decimal("0"))

    # Calculate noise ratio using only historical bars
    noise_series = calculate_noise_ratio(bars)
    noise_sma = noise_series.tail(lookback).mean()

    # Use most recent completed bar for range
    prev_bar = bars.iloc[-1]
    prev_range = prev_bar["high"] - prev_bar["low"]

    breakout_step = prev_range * noise_sma
    return Price(Decimal(str(current_bar["open"])) + Decimal(str(breakout_step)))


def get_current_close(engine: "BacktestEngine", symbol: str) -> Price:
    """Get current close price."""
    bar = engine.get_bar(symbol)
    if bar is None:
        return Price(Decimal("0"))
    return Price(Decimal(str(bar["close"])))


def get_current_open(engine: "BacktestEngine", symbol: str) -> Price:
    """Get current open price."""
    bar = engine.get_bar(symbol)
    if bar is None:
        return Price(Decimal("0"))
    return Price(Decimal(str(bar["open"])))
