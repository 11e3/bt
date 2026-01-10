"""Price calculation functions."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.types import Price

# indicators 모듈 임포트
from bt.strategies.indicators import calculate_noise_ratio

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine


def get_current_close(engine: BacktestEngine, symbol: str) -> Price:
    """Returns the current bar's close price."""
    bar = engine.data_provider.get_bar(symbol)
    if bar is None:
        return Price(Decimal("0"))
    return Price(Decimal(str(bar["close"])))


def get_vbo_buy_price(engine: BacktestEngine, symbol: str) -> Price:
    """Calculate VBO breakout buy price.

    Formula: Open + (Prev Range * Avg Noise)
    """
    lookback = engine.config.lookback

    current_bar = engine.get_bar(symbol)
    if current_bar is None:
        return Price(Decimal("0"))

    # Need history: lookback (for SMA) + 1 (for prev range) + 1 (current)
    # VBO uses noise SMA of previous N days
    bars = engine.get_bars(symbol, lookback + 2)
    if bars is None or len(bars) < lookback + 1:
        return Price(Decimal("0"))

    # 1. Indicator Calculation
    noise_series = calculate_noise_ratio(bars)

    # 2. Logic: Get noise SMA (excluding current bar)
    # Using the last 'lookback' values excluding the current unfinished bar
    noise_sma = noise_series.iloc[:-1].tail(lookback).mean()

    # 3. Logic: Get previous bar's range
    prev_bar = bars.iloc[-2]
    prev_range = prev_bar["high"] - prev_bar["low"]

    # 4. Final Calculation
    breakout_step = prev_range * noise_sma
    buy_price = Decimal(str(current_bar["open"])) + Decimal(str(breakout_step))

    return Price(buy_price)
