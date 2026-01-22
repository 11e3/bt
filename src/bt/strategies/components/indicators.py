"""Technical indicator components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bt.interfaces.strategy_types import IStrategyComponent
from bt.utils.indicator_cache import get_indicator_cache

if TYPE_CHECKING:
    import pandas as pd


class BaseIndicator(IStrategyComponent):
    """Base class for technical indicators."""

    def validate(self) -> bool:
        """Validate indicator configuration."""
        return True


class SMAIndicator(BaseIndicator):
    """Simple Moving Average indicator with caching."""

    def __init__(self, lookback: int):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate SMA using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "sma", self.lookback, prices)


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average indicator with caching."""

    def __init__(self, lookback: int):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate EMA using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "ema", self.lookback, prices)


class RSIIndicator(BaseIndicator):
    """Relative Strength Index indicator with caching."""

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate RSI using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "rsi", self.lookback, prices)


class MomentumIndicator(BaseIndicator):
    """Momentum indicator for trend analysis."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate(self, prices: pd.Series) -> float:
        """Calculate momentum over lookback period."""
        if len(prices) < self.lookback + 1:
            return 0.0

        current_price = float(prices.iloc[-2])  # Previous day
        old_price = float(prices.iloc[-(self.lookback + 1)])  # Lookback days ago

        return (current_price / old_price - 1) if old_price > 0 else -999.0
