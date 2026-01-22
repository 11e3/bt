"""Pricing components for order execution price calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bt.interfaces.strategy_types import IStrategyComponent

if TYPE_CHECKING:
    import pandas as pd

    from bt.interfaces.protocols import IBacktestEngine


class BasePricing(IStrategyComponent):
    """Base class for pricing strategies."""

    def validate(self) -> bool:
        """Validate pricing configuration."""
        return True

    def get_description(self) -> str:
        """Get pricing description."""
        return f"{self.__class__.__name__}"

    def calculate_price(self, engine: IBacktestEngine, symbol: str) -> float:
        """IPricing protocol method - delegates to __call__."""
        return self(engine, symbol)


class CurrentClosePricing(BasePricing):
    """Uses current close price for execution."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["close"])


class CurrentOpenPricing(BasePricing):
    """Uses current open price for execution."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["open"])


class VolatilityBreakoutPricing(BasePricing):
    """VBO pricing - calculates breakout buy price."""

    def __init__(self, lookback: int = 5, k_factor: float = 0.5, **_kwargs):
        self.lookback = lookback
        self.k_factor = k_factor

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        lookback = self.lookback
        bars = engine.get_bars(symbol, lookback + 1)

        if bars is None or len(bars) < lookback + 1:
            return 0.0

        # Use centralized VBO price calculation
        return self._calculate_vbo_buy_price(symbol, bars)

    def _calculate_vbo_buy_price(self, _symbol: str, bars: pd.DataFrame) -> float:
        """Calculate VBO buy price."""
        lookback = self.lookback

        # Calculate range and noise
        close_prices = bars["close"].iloc[:-1]
        high_prices = bars["high"].iloc[:-1]
        low_prices = bars["low"].iloc[:-1]

        if len(close_prices) < lookback:
            return float(bars["close"].iloc[-1])

        avg_range = (high_prices - low_prices).mean()
        noise_ratio = 1.0  # Simplified

        # Calculate buy price
        last_close = float(close_prices.iloc[-1])
        noise_adjusted_range = avg_range * noise_ratio
        return last_close + noise_adjusted_range * self.k_factor


class VBOPortfolioPricing(BasePricing):
    """VBO Portfolio pricing - calculates target buy price.

    Target price = Open + (Prev High - Prev Low) * noise_ratio

    Note: This returns the raw target price without slippage.
    Slippage is applied by portfolio.buy() during execution.
    """

    def __init__(self, noise_ratio: float = 0.5, **_kwargs):
        self.noise_ratio = noise_ratio

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return 0.0

        # Get previous bar for range calculation
        bars = engine.get_bars(symbol, 2)
        if bars is None or len(bars) < 2:
            return float(current_bar["close"])

        prev_high = float(bars["high"].iloc[-2])
        prev_low = float(bars["low"].iloc[-2])
        current_open = float(current_bar["open"])

        # Target price = open + (prev_high - prev_low) * noise_ratio
        return current_open + (prev_high - prev_low) * self.noise_ratio
