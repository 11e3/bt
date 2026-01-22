"""Condition components for buy/sell signal generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bt.interfaces.strategy_types import IStrategyComponent
from bt.utils.indicator_cache import get_indicator_cache

if TYPE_CHECKING:
    import pandas as pd

    from bt.interfaces.protocols import IBacktestEngine


class BaseCondition(IStrategyComponent):
    """Base class for trading conditions."""

    def validate(self) -> bool:
        """Validate condition configuration."""
        return True

    def get_description(self) -> str:
        """Get condition description."""
        return f"{self.__class__.__name__}"

    def evaluate(self, engine: IBacktestEngine, symbol: str) -> bool:
        """ICondition protocol method - delegates to __call__."""
        return self(engine, symbol)


class NoOpenPositionCondition(BaseCondition):
    """True when no open position exists."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        position = engine.portfolio.get_position(symbol)
        return not position.is_open


class PriceAboveSMACondition(BaseCondition):
    """True when price is above SMA."""

    def __init__(self, lookback: int, use_current_bar: bool = False):
        self.lookback = lookback
        self.use_current_bar = use_current_bar

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        bars = engine.get_bars(symbol, self.lookback + 1)
        if bars is None or len(bars) < self.lookback + 1:
            return False

        # Use cached SMA calculation
        cache = get_indicator_cache()
        close_series = bars["close"].iloc[:-1] if not self.use_current_bar else bars["close"]
        sma_value = cache.calculate_indicator(symbol, "sma", self.lookback, close_series)

        current_price = float(bars["close"].iloc[-1])
        return current_price > sma_value


class VolatilityBreakoutCondition(BaseCondition):
    """VBO breakout condition - price above volatility threshold."""

    def __init__(self, k_factor: float = 0.5, lookback: int = 5, **_kwargs):
        self.k_factor = k_factor
        self.lookback = lookback

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Calculate buy price (using centralized pricing component)
        buy_price = self._calculate_vbo_buy_price(engine, symbol)
        current_high = float(current_bar["high"])

        return current_high >= buy_price

    def _calculate_vbo_buy_price(self, engine: IBacktestEngine, symbol: str) -> float:
        """Calculate VBO buy price using centralized logic."""
        lookback = self.lookback
        bars = engine.get_bars(symbol, lookback + 1)

        if bars is None or len(bars) < lookback + 1:
            return 0.0

        # Calculate noise ratio
        close_prices = bars["close"].iloc[:-1]  # Exclude current bar
        high_prices = bars["high"].iloc[:-1]
        low_prices = bars["low"].iloc[:-1]

        if len(close_prices) < lookback:
            return 0.0

        # Range calculation
        avg_range = np.mean(high_prices - low_prices)

        # Calculate noise ratio
        noise_ratio = self._calculate_noise_ratio(close_prices)

        # VBO buy price calculation
        last_close = float(close_prices.iloc[-1])
        noise_adjusted_range = avg_range * noise_ratio
        return last_close + noise_adjusted_range * self.k_factor

    def _calculate_noise_ratio(self, prices: pd.Series) -> float:
        """Calculate noise ratio for volatility adjustment.

        Noise ratio = sum of absolute daily changes / total price range
        Higher value indicates choppy/noisy market, lower indicates trending.
        """
        if len(prices) < 2:
            return 1.0

        # Sum of absolute daily changes
        changes = prices.diff().abs()
        total_abs_changes = changes.sum()

        # Total price range
        total_range = abs(prices.iloc[-1] - prices.iloc[0])

        if total_range == 0:
            return 1.0

        # Calculate and clamp to reasonable range
        noise_ratio = total_abs_changes / total_range
        return float(min(max(noise_ratio, 0.1), 2.0))


class VBOPortfolioBuyCondition(BaseCondition):
    """VBO Portfolio buy condition with BTC market filter.

    Buy signal when:
    - Current high >= target price (open + range * noise_ratio)
    - Previous close > Previous MA5
    - Previous BTC close > Previous BTC MA20
    """

    def __init__(
        self,
        ma_short: int = 5,
        btc_ma: int = 20,
        noise_ratio: float = 0.5,
        btc_symbol: str = "BTC",
        **_kwargs,
    ):
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.noise_ratio = noise_ratio
        self.btc_symbol = btc_symbol

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get current bar
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Get historical bars for this symbol
        # Need ma_short + 1 bars: ma_short for MA calculation + 1 current bar (which gets excluded)
        bars = engine.get_bars(symbol, self.ma_short + 1)
        if bars is None or len(bars) < self.ma_short + 1:
            return False

        # Calculate previous values (excluding current bar)
        prev_close = float(bars["close"].iloc[-2])
        prev_high = float(bars["high"].iloc[-2])
        prev_low = float(bars["low"].iloc[-2])

        # Calculate MA5 on previous closes (excluding current bar)
        close_series = bars["close"].iloc[:-1]
        if len(close_series) < self.ma_short:
            return False
        prev_ma5 = float(close_series.iloc[-self.ma_short :].mean())

        # Check coin trend condition: prev_close > prev_ma5
        if prev_close <= prev_ma5:
            return False

        # Get BTC data for market filter
        # Need btc_ma + 1 bars: btc_ma for MA calculation + 1 current bar (which gets excluded)
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 1)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 1:
            return False

        # Calculate BTC previous values
        btc_close_series = btc_bars["close"].iloc[:-1]
        if len(btc_close_series) < self.btc_ma:
            return False
        prev_btc_close = float(btc_close_series.iloc[-1])
        prev_btc_ma20 = float(btc_close_series.iloc[-self.btc_ma :].mean())

        # Check BTC market condition: prev_btc_close > prev_btc_ma20
        if prev_btc_close <= prev_btc_ma20:
            return False

        # Calculate target price: open + (prev_high - prev_low) * noise_ratio
        current_open = float(current_bar["open"])
        target_price = current_open + (prev_high - prev_low) * self.noise_ratio

        # Check breakout condition: current high >= target price
        current_high = float(current_bar["high"])
        return current_high >= target_price


class VBOPortfolioSellCondition(BaseCondition):
    """VBO Portfolio sell condition.

    Sell signal when:
    - Previous close < Previous MA5 OR
    - Previous BTC close < Previous BTC MA20
    """

    def __init__(
        self,
        ma_short: int = 5,
        btc_ma: int = 20,
        btc_symbol: str = "BTC",
        **_kwargs,
    ):
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.btc_symbol = btc_symbol

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get historical bars for this symbol
        # Need ma_short + 1 bars: ma_short for MA calculation + 1 current bar (which gets excluded)
        bars = engine.get_bars(symbol, self.ma_short + 1)
        if bars is None or len(bars) < self.ma_short + 1:
            return False

        # Calculate previous values (excluding current bar)
        close_series = bars["close"].iloc[:-1]
        if len(close_series) < self.ma_short:
            return False
        prev_close = float(close_series.iloc[-1])
        prev_ma5 = float(close_series.iloc[-self.ma_short :].mean())

        # Check coin trend exit condition
        coin_sell_signal = prev_close < prev_ma5

        # Get BTC data for market filter
        # Need btc_ma + 1 bars: btc_ma for MA calculation + 1 current bar (which gets excluded)
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 1)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 1:
            return coin_sell_signal  # If no BTC data, rely on coin signal only

        # Calculate BTC previous values
        btc_close_series = btc_bars["close"].iloc[:-1]
        if len(btc_close_series) < self.btc_ma:
            return coin_sell_signal
        prev_btc_close = float(btc_close_series.iloc[-1])
        prev_btc_ma20 = float(btc_close_series.iloc[-self.btc_ma :].mean())

        # Check BTC market exit condition
        btc_sell_signal = prev_btc_close < prev_btc_ma20

        # Sell if either condition is met
        return coin_sell_signal or btc_sell_signal
