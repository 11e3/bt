"""Strategy building blocks - reusable components.

Consolidated allocation, conditions, pricing, and indicators
from scattered strategy files into organized components.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from bt.interfaces.strategy_types import IAllocation, ICondition, IPricing, IStrategyComponent
from bt.utils.decimal_cache import get_decimal
from bt.utils.indicator_cache import get_indicator_cache

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


# === ALLOCATION COMPONENTS ===


class BaseAllocation(IStrategyComponent):
    """Base class for allocation strategies."""

    def __init__(self, **config):
        self.config = config
        self.validate()

    def validate(self) -> bool:
        """Validate allocation configuration."""
        return True

    def get_description(self) -> str:
        """Get allocation description."""
        return f"{self.__class__.__name__}({self.config})"


class AllInAllocation(BaseAllocation):
    """Buy with all available cash accounting for costs."""

    def __call__(self, engine: "IBacktestEngine", _symbol: str, price: float) -> float:
        if engine.portfolio.cash <= 0 or price <= 0:
            return 0.0

        # Calculate cost multiplier (1 + fee + slippage)
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        max_affordable = float(engine.portfolio.cash) / (price * cost_multiplier)

        return max_affordable * 0.99999  # Safety buffer


class EqualWeightAllocation(BaseAllocation):
    """Equal weight allocation across all symbols."""

    def __call__(self, engine: "IBacktestEngine", _symbol: str, price: float) -> float:
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class MomentumAllocation(BaseAllocation):
    """Momentum allocation - equal weight allocation with momentum filter."""

    def __call__(self, engine: "IBacktestEngine", symbol: str, price: float) -> float:
        mom_lookback = self.config.get("mom_lookback", 20)

        # Check momentum for this symbol
        bars = engine.get_bars(symbol, mom_lookback + 2)
        if bars is None or len(bars) < mom_lookback + 2:
            return 0.0

        close_prices = bars["close"].values
        prev_close = close_prices[-2]
        old_close = close_prices[-(mom_lookback + 2)]

        momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
        if np.isnan(momentum) or momentum <= 0:
            return 0.0

        # Equal allocation among symbols
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class VolatilityBreakoutAllocation(BaseAllocation):
    """VBO momentum allocation - allocate to top N momentum assets equally."""

    def __call__(self, engine: "IBacktestEngine", symbol: str, price: float) -> float:
        top_n = self.config.get("top_n", 3)
        mom_lookback = self.config.get("mom_lookback", 20)

        # Vectorized momentum calculation
        momentum_data = {}
        all_symbols = engine.data_provider.symbols

        for s in all_symbols:
            bars = engine.get_bars(s, mom_lookback + 2)
            if bars is not None and len(bars) >= mom_lookback + 2:
                close_prices = bars["close"].values
                prev_close = close_prices[-2]
                old_close = close_prices[-(mom_lookback + 2)]

                momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
                momentum_data[s] = momentum if not np.isnan(momentum) else -999.0
            else:
                momentum_data[s] = -999.0

        # Get top symbols
        sorted_items = sorted(momentum_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in sorted_items[:top_n]]

        if symbol not in top_symbols:
            return 0.0

        # Equal allocation among top symbols
        total_equity = float(engine.portfolio.value)
        target_amount = total_equity / top_n

        cash = float(engine.portfolio.cash)
        buy_amount = min(target_amount, cash * 0.999)  # Safety buffer

        if buy_amount <= 0:
            return 0.0

        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return buy_amount / (price * cost_multiplier)


# === CONDITION COMPONENTS ===


class BaseCondition(IStrategyComponent):
    """Base class for trading conditions."""

    def validate(self) -> bool:
        """Validate condition configuration."""
        return True


class NoOpenPositionCondition(BaseCondition):
    """True when no open position exists."""

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
        position = engine.portfolio.get_position(symbol)
        return not position.is_open


class PriceAboveSMACondition(BaseCondition):
    """True when price is above SMA."""

    def __init__(self, lookback: int, use_current_bar: bool = False):
        self.lookback = lookback
        self.use_current_bar = use_current_bar

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
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

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Calculate buy price (using centralized pricing component)
        buy_price = self._calculate_vbo_buy_price(engine, symbol)
        current_high = float(current_bar["high"])

        return current_high >= buy_price

    def _calculate_vbo_buy_price(self, engine: "IBacktestEngine", symbol: str) -> float:
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
        """Calculate noise ratio for volatility adjustment."""
        if len(prices) < 2:
            return 1.0

        # Price changes
        changes = prices.diff().abs()
        avg_change = changes.mean()

        return 1.0 if avg_change == 0 else 1.0


# === PRICING COMPONENTS ===


class BasePricing(IStrategyComponent):
    """Base class for pricing strategies."""

    def validate(self) -> bool:
        """Validate pricing configuration."""
        return True


class CurrentClosePricing(BasePricing):
    """Uses current close price for execution."""

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["close"])


class CurrentOpenPricing(BasePricing):
    """Uses current open price for execution."""

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["open"])


class VolatilityBreakoutPricing(BasePricing):
    """VBO pricing - calculates breakout buy price."""

    def __init__(self, lookback: int = 5, k_factor: float = 0.5, **_kwargs):
        self.lookback = lookback
        self.k_factor = k_factor

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> float:
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

        avg_range = np.mean(high_prices - low_prices)
        noise_ratio = 1.0  # Simplified

        # Calculate buy price
        last_close = float(close_prices.iloc[-1])
        noise_adjusted_range = avg_range * noise_ratio
        return last_close + noise_adjusted_range * self.k_factor


# === INDICATOR COMPONENTS ===


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


# === VBO PORTFOLIO COMPONENTS ===


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

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
        # Get current bar
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Get historical bars for this symbol
        bars = engine.get_bars(symbol, self.ma_short + 2)
        if bars is None or len(bars) < self.ma_short + 2:
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
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 2)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 2:
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

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> bool:
        # Get historical bars for this symbol
        bars = engine.get_bars(symbol, self.ma_short + 2)
        if bars is None or len(bars) < self.ma_short + 2:
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
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 2)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 2:
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


class VBOPortfolioPricing(BasePricing):
    """VBO Portfolio pricing - calculates target buy price.

    Target price = Open + (Prev High - Prev Low) * noise_ratio
    """

    def __init__(self, noise_ratio: float = 0.5, **_kwargs):
        self.noise_ratio = noise_ratio

    def __call__(self, engine: "IBacktestEngine", symbol: str) -> float:
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


class VBOPortfolioAllocation(BaseAllocation):
    """VBO Portfolio allocation - equal weight among all portfolio coins (1/N).

    Allocates total_equity / n_strategies to each coin.
    """

    def __call__(self, engine: "IBacktestEngine", _symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0

        # Get number of symbols (n_strategies)
        n_strategies = len(engine.data_provider.symbols)
        if n_strategies == 0:
            return 0.0

        # Calculate target allocation: total_equity / n_strategies
        total_equity = float(engine.portfolio.value)
        target_alloc = total_equity / n_strategies

        # Limit to available cash
        cash = float(engine.portfolio.cash)
        buy_value = min(target_alloc, cash * 0.99)  # 99% safety buffer

        if buy_value <= 0:
            return 0.0

        # Account for fees and slippage
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return buy_value / (price * cost_multiplier)


# === FACTORY FUNCTIONS ===


def create_allocation(allocation_type: str, **config) -> IAllocation:
    """Factory function for allocation strategies."""

    allocations = {
        "all_in": AllInAllocation,
        "equal_weight": EqualWeightAllocation,
        "equal_weight_momentum": MomentumAllocation,
        "volatility_breakout": VolatilityBreakoutAllocation,
        "vbo_portfolio": VBOPortfolioAllocation,
    }

    if allocation_type not in allocations:
        raise ValueError(f"Unknown allocation type: {allocation_type}")

    return allocations[allocation_type](**config)


def create_condition(condition_type: str, **config) -> ICondition:
    """Factory function for condition strategies."""

    conditions = {
        "no_open_position": NoOpenPositionCondition,
        "price_above_sma": PriceAboveSMACondition,
        "volatility_breakout": VolatilityBreakoutCondition,
        "vbo_portfolio_buy": VBOPortfolioBuyCondition,
        "vbo_portfolio_sell": VBOPortfolioSellCondition,
    }

    if condition_type not in conditions:
        raise ValueError(f"Unknown condition type: {condition_type}")

    return conditions[condition_type](**config)


def create_pricing(pricing_type: str, **config) -> IPricing:
    """Factory function for pricing strategies."""

    pricing_strategies = {
        "current_close": CurrentClosePricing,
        "current_open": CurrentOpenPricing,
        "volatility_breakout": VolatilityBreakoutPricing,
        "vbo_portfolio": VBOPortfolioPricing,
    }

    if pricing_type not in pricing_strategies:
        raise ValueError(f"Unknown pricing type: {pricing_type}")

    return pricing_strategies[pricing_type](**config)


def create_indicator(indicator_type: str, **config) -> BaseIndicator:
    """Factory function for technical indicators."""

    indicators = {
        "sma": SMAIndicator,
        "ema": EMAIndicator,
        "rsi": RSIIndicator,
        "momentum": MomentumIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    return indicators[indicator_type](**config)
